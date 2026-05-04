# AI Runner

Simple Python scripts for local AI experiments.

## Scripts

| Name | Script | Description | Link |
|:-------------:|:-------------:|:-------------:|:-------------:|
| mms | mms-text-to-speech.py | Massively Multilingual Speech (MMS): Portuguese Text-to-Speech | [HuggingFace](https://huggingface.co/facebook/mms-tts-por)
| parler | parler-text-to-speech.py | Parler-TTS Mini Multilingual v1.1 | [HuggingFace](https://huggingface.co/parler-tts/parler-tts-mini-multilingual-v1.1)
| demucs | drumless-music.py | Demucs v4 local music source separation for generating drumless MP3 tracks; supports htdemucs, htdemucs_ft, and mdx_extra  | [GitHub](https://github.com/facebookresearch/demucs)


## Setup

Install the requirements based on the script you want to run. Change MMS to the script name you want and run using either CPU or GPU (With CUDA)

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
-- CPU
pip install -r requirements-mms.txt
-- GPU (Cuda)
pip install -r requirements-mms-cuda.txt
```

## Run

Run the script you want with ```python SCRIPT.py```.

## Notes

- The first run may download model files from Hugging Face.
- If CUDA is configured correctly, the script can run on your NVIDIA GPU.