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


def mlsmake_parser(subparser):
    subparser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory where product data is located"
    )
    subparser.add_argument(
        "--make",
        action="store_true",
        help="Whether to create a file for the product"
    )
    subparser.add_argument(
        "--radii",
        type=int,
        default=200,
        help="Whether to create a file for the product"
    )


def screening_parser(subparser):
    subparser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Path to file for wich data will be screened"
    )
    subparser.add_argument(
        "--screen-file",
        type=str,
        default=None,
        help="Path to the screening file"
    )
