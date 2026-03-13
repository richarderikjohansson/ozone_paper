import numpy as np

file = (
    "/home/ric/Software/github/op/scripts/timeseries/data/matching/mira2_matching.npy"
)
m2 = np.load(file, allow_pickle=True).item()


def mk_grid(m2):
    pgrid = np.array([val["pgrid"] for val in m2.values()])
    apriori = np.array([val["apriori"] for val in m2.values()])

    np.save(
        file="/home/ric/Software/github/op/scripts/timeseries/data/matching/pgrid.npy",
        arr=pgrid[0],
    )
    np.save(
        file="/home/ric/Software/github/op/scripts/timeseries/data/matching/apriori.npy",
        arr=apriori[0],
    )


mk_grid(m2)
