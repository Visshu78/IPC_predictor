import fasttext
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ipc import find_ipc_hybrid

# Load fastText language detector
lang_model = fasttext.load_model("lid.176.bin")

# Load NLLB-200 translator
nllb_model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(nllb_model_name)
translator = AutoModelForSeq2SeqLM.from_pretrained(nllb_model_name)

def detect_lang(text):
    label, prob = lang_model.predict(text)
    lang_code = label[0].replace("__label__", "")
    return lang_code, prob

def translate(text, target_lang):
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = translator.generate(
        **inputs,
        forced_bos_token_id=tokenizer.lang_code_to_id[target_lang]
    )
    return tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

def process_input(user_text):
    # 1. Detect language
    src_lang, prob = detect_lang(user_text)

    # Map fastText → NLLB codes
    lang_map = {
        "en": "eng_Latn",
        "hi": "hin_Deva",
        "te": "tel_Telu",
        "ta": "tam_Taml",
        "ml": "mal_Mlym",
        "kn": "kan_Knda",
        "mr": "mar_Deva",
        "bn": "ben_Beng",
        "gu": "guj_Gujr",
        "pa": "pan_Guru",
        "or": "ori_Orya",
        "ur": "urd_Arab"
    }

    # If already English, skip translation
    if src_lang == "en":
        eng_text = user_text
    else:
        eng_text = translate(user_text, "eng_Latn")

    # 2. Run IPC classifier (your existing function)
    ipc_results = find_ipc_hybrid(eng_text, top_k=3)

    # 3. Translate back results to user’s original language
    final_results = []
    for r in ipc_results:
        result_text = f"Section {r['IPC']}: {r['offense']} | Punishment: {r['punishment']}"
        if src_lang != "en":
            result_text = translate(result_text, lang_map[src_lang])
        final_results.append(result_text)

    return final_results
