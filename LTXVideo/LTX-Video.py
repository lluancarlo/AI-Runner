import torch
from diffusers import LTXPipeline
from diffusers.utils import export_to_video

# --- Configuration ---
MODEL_ID = "Lightricks/LTX-Video"
OUTPUT_FILE = "output.mp4"

PROMPT = (
    "Morning scene inside a small Italian coffee shop. "
    "Two people sit at a wooden table near the window, talking casually and drinking espresso. "
    "Warm sunlight enters from the street, casting soft shadows on the floor and table. "
    "The background shows a bar counter with a coffee machine and cups on shelves. "
    "Natural, realistic colors and lighting, subtle reflections on the cups. "
    "Static medium shot at eye level, no fast camera movement."
)

NEGATIVE_PROMPT = (
    "worst quality, low quality, inconsistent motion, jittery, distorted, warped faces, "
    "text, subtitles, logos, extra limbs, glitch, noisy background"
)

# # Light HD-ish
# WIDTH = 1216
# HEIGHT = 704 
# NUM_FRAMES = 97
# # Near Full HD
# WIDTH = 1856   # 58 * 32  (instead of 1920)
# HEIGHT = 1056  # 33 * 32  (instead of 1080)
# NUM_FRAMES = 161  # 8 * 20 + 1
# # 2K-ish
# WIDTH = 1984   # 62 * 32 (close to 2048)
# HEIGHT = 1056  # 33 * 32
# NUM_FRAMES = 161  # 8 * 20 + 1
# # 4K-ish
# WIDTH = 3808   # 119 * 32 (close to 3840)
# HEIGHT = 2144  # 67 * 32  (close to 2160)
# NUM_FRAMES = 161  # or 257, but 161 is saner
WIDTH = 1216
HEIGHT = 704
NUM_FRAMES = 257          # must be (N * 8) + 1, e.g. 97, 161, 257

NUM_STEPS = 30
GUIDANCE_SCALE = 3.5
DECODE_TIMESTEP = 0.03
DECODE_NOISE_SCALE = 0.025
FPS = 30
SEED = 42
# ---------------------


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires an NVIDIA GPU.")

    device = "cuda"
    dtype = torch.bfloat16

    print(f"Using device: {device}")
    print(f"Loading model: {MODEL_ID}")

    pipe = LTXPipeline.from_pretrained(MODEL_ID, torch_dtype=dtype)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    generator = torch.Generator(device=device).manual_seed(SEED)

    print("Generating video...")
    video = pipe(
        prompt=PROMPT,
        negative_prompt=NEGATIVE_PROMPT,
        width=WIDTH,
        height=HEIGHT,
        num_frames=NUM_FRAMES,
        num_inference_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        decode_timestep=DECODE_TIMESTEP,
        decode_noise_scale=DECODE_NOISE_SCALE,
        generator=generator,
    ).frames[0]

    print(f"Saving video to: {OUTPUT_FILE}")
    export_to_video(video, OUTPUT_FILE, fps=FPS)
    print("Done.")


if __name__ == "__main__":
    main()