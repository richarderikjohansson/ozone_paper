# from pathlib import Path
from ._const import cli_commands
from .parsers import arts_parser, m2make_parser
import argparse


def cli():
    commands, descs = cli_commands()
    parser = argparse.ArgumentParser(add_help=True)
    subparsers = parser.add_subparsers(
        dest="command", required=True, description="Available commands"
    )

    for command in commands:
        desc = descs[command]
        subparser = subparsers.add_parser(command, description=desc)

        match command:
            case "arts":
                arts_parser(subparser)
            case "m2make":
                m2make_parser(subparser)

    args = parser.parse_args()

    match args.command:
        case "arts":
            arts = commands[args.command]
            arts(
                start=args.start,
                end=args.end,
                nf=args.nf,
                summer=args.summer,
                save=args.save,
            )

        case "m2make":
            m2make = commands[args.command]
            m2make(root=args.root, make=args.make)
