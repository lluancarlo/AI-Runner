# AI Runner

Simple Python scripts for local AI experiments.

Currently included:
- `TextToSpeech.py` — generates Portuguese speech using `facebook/mms-tts-por` from Hugging Face.

## Setup

```powershell
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

### TextToSpeech.py

```powershell
python TextToSpeech.py
```

## Notes

- The first run may download model files from Hugging Face.
- If CUDA is configured correctly, the script can run on your NVIDIA GPU.