TRANSLATION_PROMPTS = {
    "almar": "Translate this from {src_lang} to {tgt_lang}:\n{src_lang}: {src_sentence} \n{tgt_lang}: ".format,
    "tower": "<|im_start|>user\n"
             "Translate the following text from {src_lang} into {tgt_lang}.\n"
             "{src_lang}: {src_sentence}.\n"
             "{tgt_lang}:<|im_end|>\n"
             "<|im_start|>assistant\n".format,
    "llama": "{src_lang}: {src_sentence} {tgt_lang}: ".format,  # although LLaMA wasn't fine-tuned for translation
    "llama2": "{src_lang}: {src_sentence} {tgt_lang}: ".format,
    "llama3": "Translate this from {src_lang} to {tgt_lang}:\n{src_lang}: {src_sentence} \n{tgt_lang}: ".format,
    "llama3.1": "Translate this from {src_lang} to {tgt_lang}:\n{src_lang}: {src_sentence} \n{tgt_lang}: ".format,
    "llama3cpo": "Translate this from {src_lang} to {tgt_lang}:\n{src_lang}: {src_sentence} \n{tgt_lang}: ".format
}

MODEL_NAME_MAP = {
    "Llama_2_13b_hf": "llama2_13B",
    "Llama_2_7b_hf": "llama2_7B",
    "ALMA_13B_R": "almar_13B",
    "ALMA_7B_R": "almar_7B",
    "ALMA_13B": "alma_13B",
    "ALMA_7B": "alma_7B",
    "TowerInstruct_13B_v0.1": "tower_13B",
    "TowerInstruct_7B_v0.1": "tower_7B",
    "Meta_Llama_3_8B": "llama3_8B",
    "Meta_Llama_3.1_8B": "llama3.1_8B",
    "Meta_Llama_3.1_70B": "llama3.1_70B",
    "Llama_3_Instruct_8B_CPO": "llama3cpo_8B"
}