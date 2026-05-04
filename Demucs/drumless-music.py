import os
import subprocess
import tempfile
import torch
import torchaudio
import imageio_ffmpeg
import soundfile as sf
from pathlib import Path
from demucs.pretrained import get_model
from demucs.apply import apply_model


# ─── config ───────────────────────────────────────────────────────────────────
INPUT_DIR   = "./input-songs"  # folder with the input .mp3 files
OUTPUT_DIR  = "./output-songs" # folder where drumless .mp3 files will be saved
MODEL_NAME  = "htdemucs_ft"    # htdemucs | htdemucs_ft | mdx_extra
MP3_BITRATE = "320k"           # Output MP3 bitrate


# ─── helpers ──────────────────────────────────────────────────────────────────
def load_mp3(path: Path, ffmpeg: str) -> tuple[torch.Tensor, int]:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name


    try:
        subprocess.run([
            ffmpeg,
            "-y",
            "-i", str(path),
            "-ar", "44100",
            "-ac", "2",
            tmp_wav
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


        data, sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
        wav = torch.from_numpy(data.T)
        return wav, sr
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)


def save_mp3(path: Path, wav: torch.Tensor, sample_rate: int, ffmpeg: str, bitrate: str):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_wav = tmp.name


    try:
        sf.write(tmp_wav, wav.T.contiguous().numpy(), sample_rate, format="WAV")


        subprocess.run([
            ffmpeg,
            "-y",
            "-i", tmp_wav,
            "-codec:a", "libmp3lame",
            "-b:a", bitrate,
            str(path)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    finally:
        if os.path.exists(tmp_wav):
            os.unlink(tmp_wav)



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using: {device.upper()}")


    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    print(f"FFmpeg: {ffmpeg_exe}")


    os.makedirs(OUTPUT_DIR, exist_ok=True)


    input_root = Path(INPUT_DIR)
    output_root = Path(OUTPUT_DIR)
    audio_files = sorted(input_root.rglob("*.mp3"))


    if not audio_files:
        print(f"No .mp3 files found in '{INPUT_DIR}'. Add your songs there and run again.")
        raise SystemExit(0)


    print(f"Loading model '{MODEL_NAME}'...")
    model = get_model(MODEL_NAME)
    model.to(device)
    model.eval()


    print(f"\nFound {len(audio_files)} file(s) to process.\n")


    for audio_path in audio_files:
        relative_path = audio_path.relative_to(input_root)
        print(f"Processing: {relative_path}")
        try:
            wav, sr = load_mp3(audio_path, ffmpeg_exe)


            if sr != model.samplerate:
                resampler = torchaudio.transforms.Resample(sr, model.samplerate)
                wav = resampler(wav)


            if wav.shape[0] == 1:
                wav = wav.repeat(2, 1)


            wav = wav.to(device)


            with torch.no_grad():
                sources = apply_model(model, wav.unsqueeze(0), split=True, overlap=0.1)[0]


            # drumless = sum of all stems EXCEPT drums
            drumless = torch.zeros_like(sources[0])
            for i, stem in enumerate(model.sources):
                if stem != "drums":
                    drumless += sources[i]


            drumless = drumless.cpu()


            out_dir = output_root / relative_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            out_mp3 = out_dir / f"{audio_path.stem} [drumless].mp3"
            save_mp3(out_mp3, drumless, model.samplerate, ffmpeg_exe, MP3_BITRATE)


            print(f"  ✓ Saved: {out_mp3}\n")


        except Exception as e:
            print(f"  ✗ Error on '{relative_path}': {e}\n")


    print("Done! All files processed.")