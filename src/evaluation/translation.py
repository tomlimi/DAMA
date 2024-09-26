import os

import torch
from datasets import load_dataset
from sacrebleu import corpus_bleu, corpus_chrf
from tqdm import tqdm
import re
from typing import List

import langcodes
from evaluation.evaluate import Evaluate
from utils.model_utils import TRANSLATION_PROMPTS
from utils.globals import *


class EvaluateTranslation(Evaluate):

    batch_size = 4

    def __init__(self, model, tok, test_file, task, model_name):

        super().__init__(model,tok, test_file, task)
        assert task == "translation", f"Task class mismatch:, expected 'translation', got '{task}' instead"
        self.model_name = model_name.split("_")[0]
        if self.model_name not in TRANSLATION_PROMPTS:
            raise ValueError(f"Model {self.model_name} is not supported for translation evaluation.")

        self.tok.padding_side = "left"
        # TODO: Unfortunatelly it won't work for Llama3 tokenzier, where add_bos_token currently has no effect
        self.tok.add_bos_token = True

        self.dataset = {"src": [], "tgt": [], "src_lang": None, "tgt_lang": None}
        self.results = {"chrf": 0., "bleu": 0., "blaser": 0.}
        self.partial_results = []

        self.load_data()

    def load_data(self):
        src_sentences = []
        tgt_sentences = []

        if self.test_file.split("_")[0] == "mt-gender":
            src_lang = "en"
            tgt_lang = self.test_file.split("_")[1]
            with open(os.path.join(DATA_DIR, "mt_gender.txt"), "r") as in_file :
                lines = in_file.readlines()
                for line in lines:
                    src_sent = line.split("\t")[2].strip()
                    src_sentences.append(src_sent)

        else: # load dataset from Hugging Face
            language_pair = self.test_file.split("_")[1]
            src_lang, tgt_lang = language_pair.split("-")
            try:
                if len(self.test_file.split("_")) > 1:
                    dataset = load_dataset(self.test_file.split("_")[0],
                                                language_pair,
                                                split='test').to_iterable_dataset()
                else:
                    dataset = load_dataset(self.test_file, split='test').to_iterable_dataset()

            except FileNotFoundError:
                raise NotImplementedError(f"The file {self.test_file} was not found in HF."
                                          f"Local test sets are not yet supported.")

            src_sentences = [te[language_pair][src_lang] for te in dataset]
            tgt_sentences = [te[language_pair][tgt_lang] for te in dataset]

        self.dataset = {"src": src_sentences, "tgt": tgt_sentences, "src_lang": src_lang, "tgt_lang": tgt_lang}

    @staticmethod
    def compute_chrf(translated_sentences: List[str], tgt_sentences: List[str]):
        return corpus_chrf(translated_sentences, [tgt_sentences]).score

    @staticmethod
    def compute_bleu(translated_sentences: List[str], tgt_sentences: List[str]):
        return corpus_bleu(translated_sentences, [tgt_sentences]).score

    def translate_sentences(self, src_sentences, src_lang, tgt_lang):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        translated_sentences = []
        translation_prompt = TRANSLATION_PROMPTS.get(self.model_name, "{src_lang}: {src_sentence} {tgt_lang}: ".format)
        prompts = [translation_prompt(src_lang=langcodes.Language(src_lang).language_name(),
                                      tgt_lang=langcodes.Language(tgt_lang).language_name(),
                                      src_sentence=sentence) for sentence in src_sentences]
        if "alma" in self.model_name:
            print("Translating with ALMA model")
        for prompt in tqdm(prompts, desc=f"Translating {src_lang} to {tgt_lang}"):

            inputs = self.tok(prompt, return_tensors="pt", padding=True, truncation=True, max_length=256).input_ids.to(device)
            # only fine-tuned model we use is ALMA
            if "alma" in self.model_name:
                with torch.no_grad():
                    generated = self.model.generate(input_ids=inputs, num_beams=5, max_new_tokens=256)
                translated = self.tok.batch_decode(generated, skip_special_tokens=True)[0].replace(prompt, "").strip()
            # For faster inference for non-fine-tuned models (translates jut till the new line marker / no
            else:
                with torch.no_grad():
                    generated = self.model.generate(input_ids=inputs, max_new_tokens=256, stop_strings=["\n"], tokenizer=self.tok)
                translated = self.tok.batch_decode(generated, skip_special_tokens=True)[0].replace(prompt, "").split("\n")[0].strip()


            translated_sentences.append(translated)
            del inputs, generated, prompt
            torch.cuda.empty_cache()

        return translated_sentences

    def evaluate(self):

        translated_sentences = self.translate_sentences(self.dataset["src"], self.dataset["src_lang"], self.dataset["tgt_lang"])

        if self.dataset["tgt"]:
            self.results["chrf"] = self.compute_chrf(translated_sentences, self.dataset["tgt"])
            self.results["bleu"] = self.compute_bleu(translated_sentences, self.dataset["tgt"])
            self.partial_results = [{"src": src, "tgt": tgt, "pred": pred} for src, tgt, pred in zip(self.dataset["src"], self.dataset["tgt"], translated_sentences)]

        else:
            self.partial_results = [{"src": src, "pred": pred} for src, pred in zip(self.dataset["src"], translated_sentences)]

