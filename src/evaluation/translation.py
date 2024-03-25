import torch
from datasets import load_dataset
from sacrebleu import BLEU, CHRF
from tqdm import tqdm
import re

from evaluation.evaluate import Evaluate


TRANSLATION_PROMPTS = {
    "almar": "Translate this from {src_lang} to {tgt_lang}:\n{src_lang}: {src_sentence} \n{tgt_lang}:".format,
    "tower": "<|im_start|>user\n"
             "Translate the following text from {src_lang} into {tgt_lang}.\n"
             "{src_lang}: {src_sentence}.\n"
             "{tgt_lang}:<|im_end|>\n"
             "<|im_start|>assistant\n".format,
    "llama": "{src_lang}: {src_sentence} {tgt_lang}:".format,  # although LLaMA wasn't fine-tuned for translation
    "llama2": "{src_lang}: {src_sentence} {tgt_lang}:".format
}

# This is exhaustive list of Tower of ALMA_R tuning languages. This needs to be extended for future.
CODE_TO_LANGUAGE = {'en': 'English', 'de': 'German', 'fr': 'French', 'es': 'Spanish', 'it': 'Italian', 'nl': 'Dutch',
                    'pt': 'Portuguese', 'ru': 'Russian', 'zh': 'Chinese', 'cs': 'Czech', 'is': 'Icelandic'}


class EvaluateTranslation(Evaluate):

    batch_size = 4

    def __init__(self, model, tok, test_file, task, model_name, with_target=False):

        super().__init__(model,tok, test_file, task)
        assert task == "translation", f"Task class mismatch:, expected 'translation', got '{task}' instead"
        self.model_name = model_name.split("_")[0]
        if self.model_name not in TRANSLATION_PROMPTS:
            raise ValueError(f"Model {self.model_name} is not supported for translation evaluation.")
        elif self.model_name.startswith("llama"):
            print(f"Model {self.model_name} is not fine-tuned for translation. Results may be poor.")
        self.tok.padding = "left"
        self.with_target = with_target

        self.results = {"chrf": 0., "bleu": 0., "blaser": 0.}
        self.partial_results = []

        self.dataset = []

        self.load_data()

    def load_data(self):
        try:
            if len(self.test_file.split("_")) > 1:
                self.dataset = load_dataset(self.test_file.split("_")[0],
                                            self.test_file.split("_")[1],
                                            split='test').to_iterable_dataset()
            else:
                self.dataset = load_dataset(self.test_file, split='test').to_iterable_dataset()

        except FileNotFoundError:
            raise NotImplementedError(f"The file {self.test_file} was not found in HF."
                                      f"Local test sets are not yet supported.")

    @staticmethod
    def compute_chrf(tgt_sentences, translated_sentences):
        chrf = CHRF()
        chrf_score = chrf.corpus_score(tgt_sentences, translated_sentences).score
        return chrf_score

    @staticmethod
    def compute_bleu(tgt_sentences, translated_sentences):
        bleu = BLEU()
        bleu_score = bleu.corpus_score(tgt_sentences, translated_sentences).score
        return bleu_score

    def translate_sentences(self, src_sentences, src_lang, tgt_lang):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        translated_sentences = []
        prompts = [TRANSLATION_PROMPTS[self.model_name](src_lang=CODE_TO_LANGUAGE[src_lang],
                                                              tgt_lang=CODE_TO_LANGUAGE[tgt_lang],
                                                              src_sentence=sentence) for sentence in src_sentences]
        # batched translation
        for bidx in tqdm(range(0, len(prompts), self.batch_size), desc="Translating batches"):
            batch = prompts[bidx:bidx+self.batch_size]

            inputs = self.tok(batch, return_tensors="pt", padding=True, truncation=True).input_ids.to(device)
            with torch.no_grad():
                generated = self.model.generate(input_ids=inputs, num_beams=5, max_new_tokens=100, do_sample=True,
                                                temperature=0.6, top_p=0.9)
            translated_batch = self.tok.batch_decode(generated, skip_special_tokens=True)
            # delete prompts from the beginning of the translation
            translated_batch = [translation.replace(prompt, "") for prompt, translation in zip(batch, translated_batch)]

            translated_sentences.extend(translated_batch)
            del inputs, generated, batch, translated_batch
            torch.cuda.empty_cache()

        return translated_sentences

    def evaluate(self):

        language_pair = self.test_file.split("_")[1]
        src_lang, tgt_lang = language_pair.split("-")

        src_sentences = [te[language_pair][src_lang] for te in self.dataset]
        translated_sentences = self.translate_sentences(src_sentences, src_lang, tgt_lang)

        if self.with_target:
            tgt_sentences = [te[language_pair][tgt_lang] for te in self.dataset]
            self.results["chrf"] = self.compute_chrf(tgt_sentences, translated_sentences)
            self.results["bleu"] = self.compute_bleu(tgt_sentences, translated_sentences)
            self.partial_results = [{"src": src, "tgt": tgt, "pred": pred} for src, tgt, pred in zip(src_sentences, tgt_sentences, translated_sentences)]

        else:
            self.partial_results = [{"src": src, "pred": pred} for src, pred in zip(src_sentences, translated_sentences)]

