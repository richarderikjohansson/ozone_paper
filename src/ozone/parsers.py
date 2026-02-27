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
        "--save", type=str, default=None, help="Name of saved simulation"
    )
    subparser.add_argument(
        "--nf", type=int, default=10000, help="Number of elements in frequency grid"
    )


def m2make_parser(subparser):
    subparser.add_argument(
        "--root", type=str, default=None, help="Root directory where all MIRA2 data is"
    )
    subparser.add_argument(
        "--make", action="store_true", help="Whether to create files for each product"
    )

    subparser.add_argument("--dataset", type=str, help="Which retrieval configuration")


def mlsmake_parser(subparser):
    subparser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory where product data is located",
    )
    subparser.add_argument(
        "--radii",
        type=int,
        default=200,
        help="Whether to create a file for the product",
    )


def screening_parser(subparser):
    subparser.add_argument(
        "--dataset", type=str, default=None, help="Dataset to perform screening on"
    )
    subparser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Name of the file with the screened data",
    )
    subparser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root directory where product data is located",
    )
    subparser.add_argument(
        "--radii",
        type=int,
        default=200,
        help="Whether to create a file for the product",
    )


def match_parser(subparser):
    subparser.add_argument(
        "--mira2", type=str, default=None, help="Path to the screened MIRA2 file"
    )
    subparser.add_argument(
        "--mls", type=str, default=None, help="Path to the screened MLS file"
    )


def plotting_parser(subparser):
    subparser.add_argument("--figure", type=str, default="all", help="Figure method")
    subparser.add_argument(
        "--filename", type=str, default=None, help="Filepath for data in figure method"
    )


def tracers_parser(subparser):
    subparser.add_argument(
        "--root",
        type=str,
        default=None,
        help="MLS directory where product data is located",
    )
