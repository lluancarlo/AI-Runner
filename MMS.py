from transformers import VitsModel, AutoTokenizer
import torch
from scipy.io.wavfile import write

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using:", device)

model_name = "facebook/mms-tts-por"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = VitsModel.from_pretrained(model_name).to(device)

text = "Olá Mariana, tudo bem? Você é muito chata"
inputs = tokenizer(text, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    output = model(**inputs).waveform

audio = output.squeeze().detach().cpu().numpy()
write("result-audio.wav", model.config.sampling_rate, audio)