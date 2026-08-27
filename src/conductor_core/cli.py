"""Click command layer for Conductor Core."""

from __future__ import annotations

import json
import sys
import traceback
from importlib.metadata import version
from pathlib import Path
from typing import Any

import click

from conductor_core import EngineConfig, GenerationRequest, LoopGenerationEngine
from conductor_core._cli import (
    EXIT_UNEXPECTED,
    EXIT_USAGE,
    CliError,
    classify_error,
    error_envelope,
    format_generation,
    format_models,
    models_envelope,
    normalize_model_records,
    should_show_progress,
    success_envelope,
)
from conductor_core.music import get_model_info
from conductor_core.storage import MAX_GENERATIONS


def _json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class _CliGroup(click.Group):
    """Render parse-time failures as JSON when machine output was requested."""

    def main(
        self,
        args: list[str] | None = None,
        prog_name: str | None = None,
        complete_var: str | None = None,
        standalone_mode: bool = True,
        **extra: Any,
    ) -> Any:
        parsed_args = list(args) if args is not None else sys.argv[1:]
        try:
            result = super().main(
                args=parsed_args,
                prog_name=prog_name,
                complete_var=complete_var,
                standalone_mode=False,
                **extra,
            )
            if standalone_mode and isinstance(result, int) and result != 0:
                raise SystemExit(result)
            return result
        except click.ClickException as exc:
            if not standalone_mode:
                raise
            if "--json" in parsed_args:
                error = CliError(EXIT_USAGE, "usage_error", exc.format_message())
                click.echo(_json_text(error_envelope(error)), err=True)
            else:
                exc.show()
            raise SystemExit(exc.exit_code) from exc
        except click.exceptions.Exit as exc:
            if not standalone_mode:
                raise
            raise SystemExit(exc.exit_code) from exc
        except click.Abort as exc:
            if not standalone_mode:
                raise
            click.echo("Aborted!", err=True)
            raise SystemExit(1) from exc


def _installed_version() -> str:
    return version("conductor-core")


def _stderr_isatty() -> bool:
    return sys.stderr.isatty()


def _emit_error(exc: BaseException, *, json_output: bool, debug: bool) -> None:
    error = classify_error(exc)
    if json_output:
        click.echo(_json_text(error_envelope(error)), err=True)
    elif debug and error.exit_code == EXIT_UNEXPECTED:
        traceback.print_exc(file=sys.stderr)
    else:
        click.echo(f"Error: {error.message}", err=True)
    raise click.exceptions.Exit(error.exit_code)


def _emit_usage_error(message: str, *, json_output: bool) -> None:
    error = CliError(EXIT_USAGE, "validation_error", message)
    if json_output:
        click.echo(_json_text(error_envelope(error)), err=True)
    else:
        click.echo(f"Error: {message}", err=True)
    raise click.exceptions.Exit(error.exit_code)


@click.group(cls=_CliGroup)
@click.version_option(version=_installed_version(), prog_name="conductor")
@click.option(
    "--debug",
    is_flag=True,
    help="Show a traceback for unexpected internal failures.",
)
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """Generate MIDI loops and inspect supported model metadata."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


@cli.command()
@click.option("--key", required=True, help="Musical key, such as C or F#.")
@click.option("--scale", required=True, help="Scale name, such as Major or Minor.")
@click.option("--description", required=True, help="Description of the desired loop.")
@click.option("--model", required=True, help="Provider model name.")
@click.option(
    "--temperature",
    type=click.FloatRange(0.0, 2.0),
    default=0.0,
    show_default=True,
    help="Sampling temperature.",
)
@click.option("--use-thinking", is_flag=True, help="Request extended reasoning.")
@click.option("--effort", help="Provider reasoning-effort value.")
@click.option("--prompt", help="Inline system-prompt override.")
@click.option(
    "--prompt-file",
    type=click.Path(exists=True, dir_okay=False, readable=True, path_type=Path),
    help="UTF-8 file containing a system-prompt override.",
)
@click.option("--render-audio", is_flag=True, help="Attempt to render an MP3.")
@click.option(
    "--soundfont-path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="SoundFont used for requested audio rendering.",
)
@click.option(
    "--artifact-root",
    type=click.Path(file_okay=False, path_type=Path),
    help="Generation-history directory.",
)
@click.option(
    "--request-timeout",
    type=click.FloatRange(min=0.0, min_open=True),
    help="Provider request timeout in seconds.",
)
@click.option(
    "--max-generations",
    type=click.IntRange(min=1),
    default=MAX_GENERATIONS,
    show_default=True,
    help="Maximum retained generations.",
)
@click.option("json_output", "--json", is_flag=True, help="Emit schema-versioned JSON.")
@click.option(
    "--quiet", is_flag=True, help="Suppress human success and progress output."
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show progress even when stderr is redirected.",
)
@click.pass_context
def generate(
    ctx: click.Context,
    key: str,
    scale: str,
    description: str,
    model: str,
    temperature: float,
    use_thinking: bool,
    effort: str | None,
    prompt: str | None,
    prompt_file: Path | None,
    render_audio: bool,
    soundfont_path: Path | None,
    artifact_root: Path | None,
    request_timeout: float | None,
    max_generations: int,
    json_output: bool,
    quiet: bool,
    verbose: bool,
) -> None:
    """Generate and persist one four-bar MIDI loop. This may incur provider cost."""
    if prompt is not None and prompt_file is not None:
        _emit_usage_error(
            "--prompt and --prompt-file are mutually exclusive.",
            json_output=json_output,
        )

    try:
        prompt_override = (
            prompt_file.read_text(encoding="utf-8")
            if prompt_file is not None
            else prompt
        )
    except (OSError, UnicodeError) as exc:
        _emit_usage_error(
            f"Could not read prompt file as UTF-8: {exc}", json_output=json_output
        )

    try:
        config = EngineConfig.from_defaults(
            artifact_root=artifact_root,
            max_generations=max_generations,
            request_timeout=request_timeout,
        )
        request = GenerationRequest(
            key=key,
            scale=scale,
            description=description,
            model=model,
            temperature=temperature,
            use_thinking=use_thinking,
            effort=effort,
            prompt_override=prompt_override,
            render_audio=render_audio,
            soundfont_path=soundfont_path,
        )
    except (TypeError, ValueError) as exc:
        _emit_usage_error(str(exc), json_output=json_output)

    progress_callback = None
    if should_show_progress(
        json_output=json_output,
        quiet=quiet,
        verbose=verbose,
        stderr_isatty=_stderr_isatty(),
    ):

        def report_progress(event: Any) -> None:
            click.echo(event.message, err=True)

        progress_callback = report_progress

    try:
        result = LoopGenerationEngine(config).generate(
            request, progress_callback=progress_callback
        )
    except Exception as exc:
        _emit_error(exc, json_output=json_output, debug=ctx.obj["debug"])

    if json_output:
        click.echo(_json_text(success_envelope(request, result)))
    elif not quiet:
        click.echo(format_generation(result))


@cli.command(name="models")
@click.option("--provider", help="Filter by provider name (case-insensitive).")
@click.option("json_output", "--json", is_flag=True, help="Emit schema-versioned JSON.")
@click.option("--quiet", is_flag=True, help="Suppress human output.")
def models_command(provider: str | None, json_output: bool, quiet: bool) -> None:
    """List packaged model metadata without contacting providers."""
    try:
        records = normalize_model_records(get_model_info(), provider)
    except ValueError as exc:
        _emit_usage_error(str(exc), json_output=json_output)

    if json_output:
        click.echo(_json_text(models_envelope(records)))
    elif not quiet:
        click.echo(format_models(records))


if __name__ == "__main__":
    cli()
