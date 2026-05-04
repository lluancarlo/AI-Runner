import torch
import soundfile as sf
from transformers import AutoTokenizer, set_seed
from parler_tts import ParlerTTSForConditionalGeneration


device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using:", device)

model_name = "parler-tts/parler-tts-mini-multilingual-v1.1"

model = ParlerTTSForConditionalGeneration.from_pretrained(model_name).to(device)

prompt_tokenizer = AutoTokenizer.from_pretrained(model_name)
description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)

text = "Olá, este é um texto convertido para áudio em português usando o Parler."
description = ( 
    "Sophia's voice speaks in European Portuguese with a clear Portugal accent, "
    "moderately slowly, with very clear audio and a natural tone."
)

description_inputs = description_tokenizer(description, return_tensors="pt")
prompt_inputs = prompt_tokenizer(text, return_tensors="pt")

description_inputs = {k: v.to(device) for k, v in description_inputs.items()}
prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}

print("Generating ...")

set_seed(42)

with torch.no_grad():
    generation = model.generate(
        input_ids=description_inputs["input_ids"],
        attention_mask=description_inputs["attention_mask"],
        prompt_input_ids=prompt_inputs["input_ids"],
        prompt_attention_mask=prompt_inputs["attention_mask"],
    )

audio = generation.squeeze().detach().cpu().numpy()
sf.write("result-parler.wav", audio, model.config.sampling_rate)

print("File 'result-parler.wav' generated successfully.")
print("Description used:", description)