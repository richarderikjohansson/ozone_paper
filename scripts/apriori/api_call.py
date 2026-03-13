import cdsapi

c = cdsapi.Client()

dataset = "reanalysis-era5-complete"
years = [
    "00",
    "01",
    "02",
    "03",
    "04",
    "05",
    "06",
    "07",
    "08",
    "09",
    "10",
    "11",
    "12",
    "13",
    "14",
    "15",
    "16",
    "17",
    "18",
    "19",
    "20",
    "21",
    "22",
    "23",
    "24",
]

for year in years:
    request = {
        "param": ["203", "152"],
        "date": f"20{year}-01-01/to/20{year}-12-31",
        "levelist": "1/to/137",
        "levtype": "ml",
        "stream": "oper",  # Denotes ERA5. Ensemble members are selected by 'enda'
        "time": "00/to/23/by/12",  # You can drop :00:00 and use MARS short-hand notation, instead of '00/06/12/18'
        "type": "an",
        "grid": "1.0/1.0",  # Latitude/longitude. Default: spherical harmonics or reduced Gaussian grid
        "area": [
            68,
            20,
            67.75,
            20.25,
        ],
        "format": "netcdf",
    }
    outfile = f"era5/era5_{year}.nc"
    print(f"Retrieving era5 data for 20{year}")
    c.retrieve(dataset, request, outfile)
