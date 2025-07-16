import sys
import argparse
import subprocess
from pathlib import Path


def run_cli(args):
    cli_path = Path(__file__).parent / "interface" / "cli.py"
    cmd = [sys.executable, str(cli_path)] + args.extra
    subprocess.run(cmd)


def run_client(args):
    client_path = Path(__file__).parent / "interface" / "client.py"
    cmd = [sys.executable, str(client_path)] + args.extra
    subprocess.run(cmd)


def run_server(args):
    server_path = Path(__file__).parent / "interface" / "server.py"
    cmd = [sys.executable, str(server_path)] + args.extra
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Transcription Pipeline Main Entry Point"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cli_parser = subparsers.add_parser("cli", help="Run CLI interface")
    cli_parser.add_argument("extra", nargs=argparse.REMAINDER, help="Arguments for CLI")
    cli_parser.set_defaults(func=run_cli)

    client_parser = subparsers.add_parser("client", help="Run API client")
    client_parser.add_argument(
        "extra", nargs=argparse.REMAINDER, help="Arguments for client"
    )
    client_parser.set_defaults(func=run_client)

    server_parser = subparsers.add_parser("server", help="Run API server")
    server_parser.add_argument(
        "extra", nargs=argparse.REMAINDER, help="Arguments for server"
    )
    server_parser.set_defaults(func=run_server)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
