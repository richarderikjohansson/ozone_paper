# from pathlib import Path
from ._const import cli_commands
from .parsers import arts_parser, m2make_parser, mlsmake_parser, screening_parser
from .logger import get_logger
from .screening import MIRA2Screener, MLSScreener
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
            case "mlsmake":
                mlsmake_parser(subparser)
            case "screen":
                screening_parser(subparser)

    args = parser.parse_args()
    logger = get_logger()

    match args.command:
        case "arts":
            arts = commands[args.command]
            arts(
                start=args.start,
                end=args.end,
                nf=args.nf,
                summer=args.summer,
                save=args.save,
                logger=logger
            )

        case "m2make":
            m2make = commands[args.command]
            m2make(root=args.root, make=args.make, logger=logger)

        case "mlsmake":
            mlsmake = commands[args.command]
            mlsmake(root=args.root, radii=args.radii, logger=logger)

        case "screen":
            if args.dataset != "mira2" and args.dataset is not None:
                mlsmake = commands["mlsmake"]
                mlsmake(root=args.root, radii=args.radii, logger=logger)
            elif args.dataset == "mira2" and args.dataset is not None:
                m2make = commands["m2make"]
                m2make(root=args.root, radii=args.radii)

            datascreen = commands[args.command]
            obj = datascreen(dataset=args.dataset, filename=args.filename)

            if obj.meta["product"] != "mira2":
                mlsscreen = MLSScreener(data=obj.data,
                                        meta=obj.meta,
                                        screen=obj.screen,
                                        logger=logger,
                                        winter=args.winter,
                                        )

                mlsscreen.save_screened_data(filename=args.filename)
            else:
                MIRA2Screener(obj.data, obj.meta, obj.screen)
