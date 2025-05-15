# diffUIsion

diffUIsion is a lightweight **FastAPI** + **Diffusers** web UI for image-generation
models (FLUX family by default).  It can optionally use a bundled local language-model to “enhance” your prompt before the image is rendered (no external API needed).

---

## Features

- One-page Bootstrap UI with live token counter
- Optional prompt enhancement via a local model
- Model/height/width/etc. taken from `models.toml` – add your own entries  
- Generation runs in an **isolated subprocess** → cancel button frees GPU RAM instantly  
- Fully reproducible (seed input)  
- Pure Python, works on CPU, CUDA, MPS & XPU

---

## Quick start

1. Clone & enter

```bash
git clone https://github.com/MattKobayashi/diffuision.git
cd diffuision
```

2. Add secrets

```bash
echo "<YOUR_HF_TOKEN>" > secrets/hf_token
```

3. Run

```bash
uv run web.py
```

Open <http://localhost:8000> in your browser.

---

## Configuration

| File / Environment Variable  | Purpose                                |
|-----------------------------|----------------------------------------|
| `secrets/hf_token`          | HuggingFace token for model downloads  |

### models.toml

Each `[[models]]` table describes overrides for height/width, number of
steps, guidance scale, etc., keyed by the HuggingFace repo id (`name`).

---

## Cancel button

Pressing **Cancel** calls `/cancel`, which terminates the child
generation process.  GPU/CPU memory is released immediately and the UI
is ready for another prompt.

---

## Folder structure

```text
├── web.py            ← main FastAPI application
├── templates/        ← Jinja2 templates (index & result)
├── generated/        ← saved images (auto-created)
├── secrets/          ← API tokens (ignored by git)
└── models.toml       ← per-model defaults
```
