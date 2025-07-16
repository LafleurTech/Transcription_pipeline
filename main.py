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
    subparsers = parser.add_subparsers(dest="command")

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

    if len(sys.argv) == 1:
        print("Select a mode to run:")
        print("1. CLI interface")
        print("2. API client")
        print("3. API server")
        choice = input("Enter choice [1-3]: ").strip()
        if choice == "1":
            sys.argv.append("cli")
        elif choice == "2":
            sys.argv.append("client")
        elif choice == "3":
            sys.argv.append("server")
        else:
            print("Invalid choice. Exiting.")
            sys.exit(1)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
