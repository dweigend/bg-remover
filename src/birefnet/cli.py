"""BiRefNet CLI - Background removal with Rich output.

LLM-friendly: Supports --json and --quiet modes for coding agents.
"""

import json
import time
from pathlib import Path
from typing import Annotated

import typer
from PIL import Image
from rich.console import Console

from . import process_image
from .model import get_device

app = typer.Typer(
    help=(
        "🖼️ BiRefNet - AI Background Removal CLI\n\n"
        "[bold]Commands:[/bold]\n"
        "  remove  Remove background from images (outputs RGBA with transparency)\n"
        "  info    Show model/device information\n\n"
        "[bold]LLM Integration:[/bold]\n"
        "  --json   Structured JSON output for parsing\n"
        "  --quiet  Paths only (for piping)\n\n"
        "[bold]Exit Codes:[/bold] 0=success, 1=error, 2=invalid args"
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)
console = Console()
error_console = Console(stderr=True)

__version__ = "0.1.0"


def version_callback(value: bool) -> None:
    """Show version and exit."""
    if value:
        console.print(f"birefnet {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            callback=version_callback,
            is_eager=True,
            help="Show version and exit.",
        ),
    ] = False,
) -> None:
    """BiRefNet Background Remover CLI."""


@app.command()
def remove(
    inputs: Annotated[
        list[Path],
        typer.Argument(
            help="Input image(s). Shell globs expanded (e.g. *.png)",
            show_default=False,
        ),
    ],
    output: Annotated[
        Path,
        typer.Option(
            "-o",
            "--output",
            help="Output directory. Created if missing",
        ),
    ] = Path("."),
    size: Annotated[
        int,
        typer.Option(
            "-s",
            "--size",
            help="Processing resolution: 512|1024|2048",
        ),
    ] = 1024,
    format_: Annotated[
        str,
        typer.Option(
            "-f",
            "--format",
            help="Output format: png|webp",
        ),
    ] = "png",
    suffix: Annotated[
        str,
        typer.Option(
            "--suffix",
            help="Appended to output filename",
        ),
    ] = "_nobg",
    quality: Annotated[
        int,
        typer.Option(
            "-q",
            "--quality",
            help="WebP quality 1-100 (png ignores this)",
            min=1,
            max=100,
        ),
    ] = 95,
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="🤖 JSON output for LLM agents",
        ),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option(
            "--quiet",
            help="📝 Paths only (for piping)",
        ),
    ] = False,
) -> None:
    """🎨 Remove background from images.

    [bold]Output:[/bold] RGBA PNG/WebP with transparent background

    [bold]Examples:[/bold]
      birefnet remove photo.jpg
      birefnet remove *.png -o out/ -s 1024
      birefnet remove img.jpg --json

    [bold]JSON Schema (--json):[/bold]
      {success: bool, processed: [{input, output, status}], failed: [], duration_ms: int}

    [bold]Quiet Mode (--quiet):[/bold]
      Outputs only paths, one per line. Exit code indicates success/failure.
    """
    start_time = time.time()
    results: list[dict] = []

    valid_inputs, failed = _validate_inputs(inputs, json_output, quiet)

    if not valid_inputs:
        _output_error(json_output, quiet, "NO_VALID_INPUTS", "No valid input files", failed)
        raise typer.Exit(1)

    output.mkdir(parents=True, exist_ok=True)

    if not json_output and not quiet:
        console.print("🔄 Loading model...")

    for idx, input_path in enumerate(valid_inputs, 1):
        try:
            out_path = _process_single_image(input_path, output, size, format_, suffix, quality)
            results.append({"input": str(input_path), "output": str(out_path), "status": "ok"})

            if quiet:
                console.print(out_path)
            elif not json_output:
                console.print(f"  [{idx}/{len(valid_inputs)}] {input_path.name} → {out_path.name} [green]✓[/green]")

        except Exception as e:
            failed.append({"input": str(input_path), "error": str(e)})
            if not json_output and not quiet:
                error_console.print(f"  [red]✗[/red] {input_path.name}: {e}")

    duration_ms = int((time.time() - start_time) * 1000)
    _output_results(results, failed, duration_ms, json_output, quiet)

    if failed:
        raise typer.Exit(1)


@app.command()
def info(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Output as JSON",
        ),
    ] = False,
) -> None:
    """Show system and model information.

    Displays: model name, compute device, version.
    Use --json for machine-readable output.
    """
    device = get_device()
    device_name = _get_device_name(device)

    if json_output:
        data = {
            "model": "ZhengPeng7/BiRefNet",
            "device": str(device),
            "device_name": device_name,
            "version": __version__,
        }
        console.print(json.dumps(data, indent=2))
    else:
        from rich.panel import Panel

        info_text = (
            f"[bold]Model:[/bold]   ZhengPeng7/BiRefNet\n"
            f"[bold]Device:[/bold]  {device} ({device_name})\n"
            f"[bold]Version:[/bold] {__version__}"
        )
        console.print(Panel(info_text, title="BiRefNet Info", border_style="blue"))


def _get_device_name(device: str) -> str:
    """Map device string to human-readable name."""
    if device == "mps":
        return "Apple Silicon"
    if device == "cuda":
        return "NVIDIA GPU"
    return "CPU"


def _validate_inputs(
    inputs: list[Path],
    json_output: bool,
    quiet: bool,
) -> tuple[list[Path], list[dict]]:
    """Filter inputs to existing files. Returns (valid, failed)."""
    valid: list[Path] = []
    failed: list[dict] = []

    for inp in inputs:
        if not inp.exists():
            failed.append({"input": str(inp), "error": "FILE_NOT_FOUND"})
            if not json_output and not quiet:
                error_console.print(f"[red]✗[/red] File not found: {inp}")
        else:
            valid.append(inp)

    return valid, failed


def _process_single_image(
    input_path: Path,
    output_dir: Path,
    size: int,
    format_: str,
    suffix: str,
    quality: int,
) -> Path:
    """Remove background from image and save. Returns output path."""
    img = Image.open(input_path)
    result_img = process_image(img, size)

    out_name = f"{input_path.stem}{suffix}.{format_}"
    out_path = output_dir / out_name

    save_kwargs: dict = {}
    if format_ == "webp":
        save_kwargs["quality"] = quality

    result_img.save(out_path, **save_kwargs)
    return out_path


def _output_results(
    results: list[dict],
    failed: list[dict],
    duration_ms: int,
    json_output: bool,
    quiet: bool,
) -> None:
    """Print summary (JSON or Rich formatted)."""
    if json_output:
        output_data = {
            "success": len(failed) == 0,
            "processed": results,
            "failed": failed,
            "duration_ms": duration_ms,
        }
        console.print(json.dumps(output_data, indent=2))
    elif not quiet:
        duration_s = duration_ms / 1000
        console.print(
            f"\n✨ Done! {len(results)} image(s) processed in {duration_s:.1f}s"
        )


def _output_error(
    json_output: bool,
    quiet: bool,
    code: str,
    message: str,
    failed: list[dict] | None = None,
) -> None:
    """Print error (JSON or Rich formatted)."""
    if json_output:
        error_data = {
            "success": False,
            "error": {"code": code, "message": message},
            "processed": [],
            "failed": failed or [],
        }
        console.print(json.dumps(error_data, indent=2))
    elif not quiet:
        error_console.print(f"[red]Error:[/red] {message}")


if __name__ == "__main__":
    app()
