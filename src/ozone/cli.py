from pathlib import Path
from ._const import cli_commands, figure_methods
from .parsers import (
    arts_parser,
    m2make_parser,
    mlsmake_parser,
    screening_parser,
    plotting_parser,
    match_parser,
)
from .logger import get_logger
from .screening import MIRA2Screener, MLSScreener
from .plotting import dynamic_caller
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
            case "match":
                match_parser(subparser)
            case "plotting":
                plotting_parser(subparser)

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
                logger=logger,
            )

        case "m2make":
            m2make = commands[args.command]
            m2make(root=args.root, make=args.make, logger=logger)

        case "mlsmake":
            mlsmake = commands[args.command]
            mlsmake(root=args.root, logger=logger)

        case "screen":
            mlsdp = ["O3", "H2O", "N2O", "ClO"]
            datascreen = commands[args.command]
            if args.dataset is None:
                logger.error("Please provide a argument for the dataset")

            if args.dataset in mlsdp:
                mlsmake = commands["mlsmake"]
                mlsmake(root=args.root, logger=logger)
                obj = datascreen(dataset=args.dataset, filename=args.filename)
                mlsscreen = MLSScreener(
                    data=obj.data,
                    meta=obj.meta,
                    screen=obj.screen,
                    logger=logger,
                    winter=True,
                )
                mlsscreen.save_screened_data(filename=args.filename)

            else:
                m2make = commands["m2make"]
                m2make(root=args.root, logger=logger, make=True, dataset=args.dataset)
                obj = datascreen(dataset=args.dataset, filename=args.filename)
                mira2screen = MIRA2Screener(
                    data=obj.data,
                    meta=obj.meta,
                    screen=obj.screen,
                    logger=logger,
                )
                mira2screen.save_screened_data(filename=args.filename)

        case "match":
            matching = commands[args.command]
            if args.mls is None or args.mira2 is None:
                return logger.error(
                    "Provide paths to screened MIRA2 and screened MLS files"
                )
            mlsfile = Path(args.mls)
            mira2file = Path(args.mira2)

            if not mlsfile.exists() or not mira2file.exists():
                logger.error("Check filepaths for the screened MIRA2 and MLS data")
            else:
                matching(mira2=mira2file, mls=mlsfile, logger=logger)

        case "plotting":
            plotting = commands[args.command]
            obj = plotting(logger=logger)

            # if args.figure == "all":
            #     methods = figure_methods()
            #     for meth, figure in methods:
            #         method = dynamic_caller(obj, meth)
            #         method(figure)

            if args.figure == "fig01":
                filename = Path(args.filename)
                name = filename.name.split(".")[0]
                assert args.figure == name
                obj.make_fig01(figure=args.figure, file=filename)
