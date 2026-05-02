# AI Runner

Simple Python scripts for local AI experiments.

## Scripts

| Name | Script | Description | Link |
|:-------------:|:-------------:|:-------------:|:-------------:|
| MMS | Simple-TextToSpeech.py | Massively Multilingual Speech (MMS): Portuguese Text-to-Speech | [HuggingFace](https://huggingface.co/facebook/mms-tts-por)

## Setup

Install the requirements based on the script you want to run. Change MMS to the script name you want

```powershell
py -3.12 -m venv .venv-MMS
.venv\Scripts\activate
pip install -r requirements-MMS.txt
```

## Run

Run the correct script inside the correct env:

```powershell
python MMS.py
```

## Notes

- The first run may download model files from Hugging Face.
- If CUDA is configured correctly, the script can run on your NVIDIA GPU.