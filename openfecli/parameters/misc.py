from plugcli.params import Option

NCORES = Option(
    "-n",
    "--n-cores",
    help="The number of cores which should be used for multiprocessing stages."
)

OVERWRITE = Option(
    "--overwrite-charges",
    is_flag=True,
    default=False,
    help="If the charges already present in the molecules should be overwritten."
)