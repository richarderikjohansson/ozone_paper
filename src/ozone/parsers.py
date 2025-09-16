def arts_parser(subparser):
    subparser.add_argument(
        "--start",
        type=float,
        default=250e9,
        help="Start frequency in Hertz",
    )
    subparser.add_argument(
        "--end",
        type=float,
        default=300e9,
        help="End frequency in Hertz",
    )
    subparser.add_argument(
        "--summer",
        action="store_true",
        help="If atmospheric conditions should reflect summer",
    )
    subparser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Name of saved simulation"
    )
    subparser.add_argument(
        "--nf",
        type=int,
        default=10000,
        help="Number of elements in frequency grid"
    )


def m2make_parser(subparser):
    subparser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory where all MIRA2 data is"
    )
    subparser.add_argument(
        "--make",
        action="store_true",
        help="Whether to create files for each product"
    )
