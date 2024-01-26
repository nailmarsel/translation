<h1>Quickstart: Run Inference</h1>
For this code, you need to install transformers==4.28.1

```
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer
)

def load_model():
    tokenizer = {}
    model = {}
    tokenizer["tat"] = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="IPSAN/nllb-200-600M-from-rus-to-tat",src_lang="tat_Cyrl")
    tokenizer["rus"] = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="IPSAN/nllb-200-600M-from-tat-to-rus", src_lang="rus_Cyrl")
    model["tat"] = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path="IPSAN/nllb-200-600M-from-rus-to-tat").to('cuda:0')
    model["rus"] = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path="IPSAN/nllb-200-600M-from-tat-to-rus").to('cuda:0')
    
    

    return model, tokenizer

def inference(sent, source_lang, model, tokenizer):
    '''
    :param sent: rus or tatar str
    :param source_lang: rus_Cyrl or tat_Cyrl
    :return:
    '''
    inputs = tokenizer[source_lang](sent, return_tensors="pt").to("cuda:7")
    translated_tokens = model[source_lang].generate(
        **inputs, forced_bos_token_id=tokenizer[source_lang].lang_code_to_id["tat_Cyrl" if source_lang == "rus" else "rus_Cyrl"], max_length=2048, num_beams=8
    )
    return tokenizer[source_lang].batch_decode(translated_tokens, skip_special_tokens=True)[0]
```
