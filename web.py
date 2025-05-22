# /// script
# requires-python = ">=3.11, <3.12"
# dependencies = [
#     "diffusers==0.33.1",
#     "fastapi[standard]==0.115.12",
#     "jinja2==3.1.6",
#     "peft==0.15.2",
#     "protobuf==6.31.0",
#     "python-multipart==0.0.20",
#     "sentencepiece==0.2.0",
#     "torch==2.7.0",
#     "transformers==4.52.3",
# ]
# ///
import asyncio
import concurrent.futures
import multiprocessing
from multiprocessing.queues import (
    Queue as MPQueue,
)
import tomllib
import uuid

from diffusers import FluxPipeline
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pathlib import Path
import torch

# Load Hugging Face token
try:
    token_path = Path("secrets/hf_token")
    hf_token = token_path.read_text().strip()
except FileNotFoundError:
    print(f"Error: The token file was not found at {token_path.resolve()}")
    print("Please create the file and add your Hugging Face token.")
    exit(1)
except Exception as e:
    print(f"An error occurred while reading the token: {e}")
    exit(1)

# Load per-model configuration from models.toml → {model_name: cfg_dict}
try:
    raw_cfg = tomllib.loads(Path("models.toml").read_text())
    MODELS_CFG = {m["name"]: m for m in raw_cfg.get("models", [])}
except FileNotFoundError:
    MODELS_CFG = {}  # fall back to hard-coded defaults if file missing

gen_process: multiprocessing.Process | None = None
gen_queue: MPQueue | None = None

_tokenizer: T5Tokenizer | None = None
_superprompt_model: T5ForConditionalGeneration | None = None


def _create_pipeline(model: str, device: torch.device, *, token: str) -> FluxPipeline:
    pipe = FluxPipeline.from_pretrained(
        model,
        torch_dtype=torch.bfloat16,
        token=token,
    )
    if "FLUX.1-dev" in model:
        pipe.load_lora_weights(
            "ByteDance/Hyper-SD",
            weight_name="Hyper-FLUX.1-dev-8steps-lora.safetensors",
            adapter_name="hyper-sd",
        )
        pipe.fuse_lora(lora_scale=0.125)
    pipe.to(device)

    # Optimisations
    pipe.enable_attention_slicing()

    return pipe


def _subproc_worker(prompt, model, token_limit, base_url, seed, enhance, q: MPQueue):
    try:
        img_path = generate_image(
            prompt,
            model,
            token_limit,
            base_url,
            seed,
            enhance=enhance,
        )
        q.put(("ok", str(img_path)))
    except Exception as e:
        q.put(("err", str(e)))


def enhance_prompt(
    short_prompt: str,
    token_limit: int,
    *,
    device: torch.device | None = None,
) -> str:
    """
    Use `roborovski/superprompt-v1` to expand `short_prompt` while keeping the
    result ≤ `token_limit` tokens (77 is the model's practical maximum).
    """
    global _tokenizer, _superprompt_model

    # Select device if not provided
    if device is None:
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.xpu.is_available():
            device = torch.device("xpu")
        else:
            device = torch.device("cpu")

    # Lazy-load model & tokenizer once
    if _tokenizer is None or _superprompt_model is None:
        _tokenizer = T5Tokenizer.from_pretrained("roborovski/superprompt-v1")
        _superprompt_model = T5ForConditionalGeneration.from_pretrained(
            "roborovski/superprompt-v1",
            device_map="auto" if device.type != "cpu" else None,
        )
        if device.type == "cpu":          # `device_map="auto"` already handles non-CPU
            _superprompt_model.to(device)

    prefix = "Expand the following prompt to add more detail: "
    input_text = prefix + short_prompt.strip()

    input_ids = _tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    # Model is trained for max 77 new tokens → respect both that and caller’s limit
    max_new = min(77, token_limit)
    output_ids = _superprompt_model.generate(input_ids, max_new_tokens=max_new)
    enhanced = _tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    # Hard trim to `token_limit` tokens (space-separated) as a safety net
    words = enhanced.split()
    if len(words) > token_limit:
        enhanced = " ".join(words[:token_limit])

    return enhanced


def generate_image(
    short_prompt: str,
    model: str,
    token_limit: int = 256,
    base_url: str | None = None,
    seed: int = 42,
    enhance: bool = True,
) -> Path:
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.xpu.is_available():
        device = torch.device("xpu")
    else:
        device = torch.device("cpu")

    cfg = MODELS_CFG.get(model, {})
    height = cfg.get("height", 1024)
    width = cfg.get("width", 1024)
    guidance_scale = cfg.get("guidance_scale", 0.0)
    num_inference_steps = cfg.get("num_inference_steps", 4)
    true_cfg_scale = cfg.get("true_cfg_scale", 1.0)
    max_sequence_length = cfg.get("max_sequence_length", 256)
    negative_prompt_cfg = cfg.get(
        "negative_prompt",
        "low quality, poor quality, ugly, deformed, bad art, poor art, tiling, watermark, "
        "text, logo, noisy, grain, artifacts, jpeg artifacts, blurry, out of focus, deformed face, "
        "bad face, ugly face, asymmetric face, distorted face, mutated face, melted face, "
        "extra eyes, misplaced eyes, crooked eyes, distorted mouth, strange mouth, weird nose, "
        "bad nose, deformed hands, mutated hands, bad hands, extra hands, missing fingers, "
        "too many fingers, distracting background, cluttered background, messy background, "
        "low detail, lack of detail, monochrome, grayscale, sketch, drawing, painting, cartoon, "
        "3d, cgi, childish, naive",
    )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        pipe_future = pool.submit(_create_pipeline, model, device, token=hf_token)

        if enhance:
            prompt_future = pool.submit(
                enhance_prompt, short_prompt, token_limit, device=device
            )
        else:
            prompt_future = None

        prompt = prompt_future.result() if enhance else short_prompt
        print("Enhanced prompt:" if enhance else "Prompt:", prompt)

        pipe = pipe_future.result()  # wait for pipeline to finish loading

    generator = torch.Generator(device=device).manual_seed(seed)

    pipe_args = dict(
        generator=generator,
        guidance_scale=guidance_scale,
        height=height,
        max_sequence_length=max_sequence_length,
        negative_prompt=negative_prompt_cfg,
        num_inference_steps=num_inference_steps,
        prompt=prompt,
        true_cfg_scale=true_cfg_scale,
        width=width,
    )

    image = pipe(**pipe_args).images[0]

    # Ensure the 'generated' directory exists
    output_dir = Path("generated")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the generated image
    image_uuid = uuid.uuid4()
    image_path = output_dir / f"image_{image_uuid}.png"
    image.save(image_path)

    return image_path


# FastAPI app and routes
app = FastAPI(title="Diffusion Model UI")

# Mount folders
app.mount("/generated", StaticFiles(directory="generated"), name="generated")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    prefill = request.query_params.get("prompt", "")
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prompt": prefill},
    )


@app.post("/generate", response_class=HTMLResponse)
async def run_generation(
    request: Request,
    prompt: str = Form(...),
    model: str = Form("black-forest-labs/FLUX.1-schnell"),
    token_limit: int = Form(256),
    base_url: str | None = Form(None),
    seed: int = Form(42),
    enhance: bool = Form(False),
):
    global gen_process, gen_queue
    ctx = multiprocessing.get_context("spawn")
    gen_queue = ctx.Queue()
    gen_process = ctx.Process(
        target=_subproc_worker,
        args=(prompt, model, token_limit, base_url, seed, enhance, gen_queue),
        daemon=True,
    )
    gen_process.start()

    try:
        status, payload = await asyncio.to_thread(
            gen_queue.get
        )  # blocking get in thread
    finally:
        gen_process.join()  # ensure reap even on error
        gen_process = None
        gen_queue = None

    if status == "err":
        return templates.TemplateResponse(
            "index.html",
            {"request": request, "error": payload},
        )

    img_path = Path(payload)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "img_url": f"/generated/{img_path.name}",
            "prompt": prompt,
        },
    )


@app.post("/cancel")
async def cancel_generation():
    global gen_process
    if gen_process is not None and gen_process.is_alive():
        gen_process.terminate()
    return {"status": "cancelling"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web:app", host="0.0.0.0", port=8000, reload=False)
