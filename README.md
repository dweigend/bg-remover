# BiRefNet

I built this CLI because cutting out images for my web and game projects was getting tedious. My goal: automate background removal via the command line so I can just tell my coding agent *"take these images and remove all backgrounds"* — and it does. Life's too short for manual masking.

Pairs well with [agent skills](https://agentskills.io/home) — whether Claude Code, OpenCode, or other coding agents. Wrap this CLI in a custom skill, and your agent handles batch processing while you focus on building. The `--json` flag makes output parseable, and skills are portable: write once, use across any compatible agent.

Powered by the [BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) model.

## Features

- 🎨 High-quality background removal with transparent output (RGBA PNG/WebP)
- ⚡ GPU acceleration (CUDA, Apple Silicon MPS) with CPU fallback
- 🤖 LLM-friendly: `--json` and `--quiet` modes for coding agents
- 📦 Batch processing with glob patterns

## Installation

```bash
# Clone and install with uv
git clone https://github.com/dweigend/bg-remover.git
cd bg-remover
uv sync

# Install development tools
uv sync --group dev
```

**Requirements:** Python 3.12+, ~2GB disk space for model weights (auto-downloaded on first run)

## Quick Start

```bash
# Remove background from a single image
uv run birefnet remove photo.jpg

# Batch process with output directory
uv run birefnet remove *.png -o output/

# Higher resolution processing
uv run birefnet remove image.jpg -s 2048

# WebP output with quality setting
uv run birefnet remove photo.jpg -f webp -q 90
```

## CLI Reference

### `birefnet remove`

Remove background from images.

```
birefnet remove [OPTIONS] INPUTS...

Arguments:
  INPUTS...              Input image(s). Shell globs expanded (e.g. *.png)

Options:
  -o, --output PATH      Output directory (default: current)
  -s, --size INT         Processing resolution: 512|1024|2048 (default: 1024)
  -f, --format TEXT      Output format: png|webp (default: png)
  --suffix TEXT          Appended to output filename (default: _nobg)
  -q, --quality INT      WebP quality 1-100 (default: 95)
  --json                 JSON output for LLM agents
  --quiet                Paths only (for piping)
```

**Examples:**

```bash
# Basic usage
birefnet remove photo.jpg
# Output: photo_nobg.png

# Custom output directory and format
birefnet remove image.png -o processed/ -f webp

# High-res processing
birefnet remove portrait.jpg -s 2048

# JSON output for automation
birefnet remove photo.jpg --json
```

### `birefnet info`

Show system and model information.

```
birefnet info [--json]
```

### Output Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| Default | Rich formatted output with progress | Interactive use |
| `--json` | Structured JSON response | LLM agents, automation |
| `--quiet` | Output paths only | Piping to other commands |

**JSON Schema (--json):**

```json
{
  "success": true,
  "processed": [{"input": "photo.jpg", "output": "photo_nobg.png", "status": "ok"}],
  "failed": [],
  "duration_ms": 1234
}
```

## Python API

```python
from birefnet import process_image
from PIL import Image

# Load and process
img = Image.open("photo.jpg")
result = process_image(img, size=1024)
result.save("photo_nobg.png")
```

**Lower-level API:**

```python
from birefnet import load_model, get_device, preprocess, infer, mask_to_pil, remove_background

device = get_device()
model = load_model(device)

processed = preprocess(img, size=1024)
mask_tensor = infer(model, processed, device)
mask_pil = mask_to_pil(mask_tensor, processed.original_size)
result = remove_background(img, mask_pil)
```

## Architecture

```
src/birefnet/
├── __init__.py      # Public API (process_image)
├── cli.py           # Typer CLI with Rich output
├── model.py         # Model loading & device detection
├── preprocess.py    # Image normalization & tensor conversion
├── inference.py     # Forward pass through BiRefNet
├── postprocess.py   # Mask → RGBA conversion
└── types.py         # Data structures (ProcessedImage)
```

**Pipeline Flow:**

```
Image → preprocess() → infer() → mask_to_pil() → remove_background() → RGBA
```

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Lint & format
uv run ruff check src/
uv run ruff format src/

# Type checking
uv run ty check src/

# CLI smoke test
uv run birefnet --help
uv run birefnet info --json
```

## License

MIT
