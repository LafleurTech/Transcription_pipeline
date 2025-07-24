import sys
import subprocess
from pathlib import Path
from typing import List, Optional
import typer
from typing_extensions import Annotated


app = typer.Typer(
    name="transcription-pipeline",
    help="Transcription Pipeline Main Entry Point",
    no_args_is_help=True,
)


def run_script(script_name: str, extra_args: List[str]):
    """Helper function to run interface scripts with extra arguments."""
    script_path = Path(__file__).parent / "interface" / script_name
    cmd = [sys.executable, str(script_path)] + extra_args
    subprocess.run(cmd)


@app.command()
def cli(
    ctx: typer.Context,
    extra_args: Annotated[Optional[List[str]], typer.Argument()] = None,
):
    """Run CLI interface."""
    args = extra_args or []
    if ctx.params.get("help"):
        args = []
    run_script("cli.py", args)


@app.command()
def client(
    ctx: typer.Context,
    extra_args: Annotated[Optional[List[str]], typer.Argument()] = None,
):
    """Run API client."""
    args = extra_args or []
    if ctx.params.get("help"):
        args = []
    run_script("client.py", args)


@app.command()
def server(
    ctx: typer.Context,
    extra_args: Annotated[Optional[List[str]], typer.Argument()] = None,
):
    """Run API server."""
    args = extra_args or []
    if ctx.params.get("help"):
        args = []
    run_script("server.py", args)


@app.command()
def streamlit(
    ctx: typer.Context,
    extra_args: Annotated[Optional[List[str]], typer.Argument()] = None,
):
    """Run Streamlit app."""
    args = extra_args or []
    if ctx.params.get("help"):
        args = []
    streamlit_path = Path(__file__).parent / "interface" / "streamlit_app.py"
    cmd = ["streamlit", "run", str(streamlit_path)] + args
    subprocess.run(cmd)


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context):
    """Interactive mode when no command is provided."""
    if ctx.invoked_subcommand is None:
        typer.echo("Select a mode to run:")
        typer.echo("1. CLI interface")
        typer.echo("2. API client")
        typer.echo("3. API server")
        typer.echo("4. Streamlit app")

        choice = typer.prompt("Enter choice [1-4]").strip()

        if choice == "1":
            ctx.invoke(cli)
        elif choice == "2":
            ctx.invoke(client)
        elif choice == "3":
            ctx.invoke(server)
        elif choice == "4":
            ctx.invoke(streamlit)
        else:
            typer.echo("Invalid choice. Exiting.")
            raise typer.Exit(1)


if __name__ == "__main__":
    app()
