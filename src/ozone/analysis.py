import numpy as np
from datetime import datetime
from datetime import time
from datetime import timedelta


def get_period(data: dict, period: str):
    rdata = {}
    match period:
        case "day":
            t0 = time(hour=10, minute=0, second=0)
            t1 = time(hour=14, minute=0, second=0)

        case "night":
            t0 = time(hour=0, minute=0, second=0)
            t1 = time(hour=4, minute=0, second=0)

    for dt, vals in data.items():
        if t0 <= dt.time() <= t1:
            rdata[dt] = vals
    return rdata


def mk_daterange(period: str):
    match period:
        case "day":
            hour = 12
        case "night":
            hour = 2

    start = datetime(
        year=2019,
        month=10,
        day=1,
        hour=hour,
        minute=0,
        second=0,
    )
    end = datetime(
        year=2020,
        month=4,
        day=30,
        hour=hour,
        minute=0,
        second=0,
    )
    daterange = []
    current = start

    while current <= end:
        daterange.append(current)
        current += timedelta(days=1)

    return np.array(daterange)
