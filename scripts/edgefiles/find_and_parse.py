from ozone.io import get_egdefiles
from ozone.utils import parse_edgefile, filter_edgedata

root = "/home/ric/Data/"
files = get_egdefiles(root)
edgefile1 = parse_edgefile(files[1])

for severity in [0, 1, 2]:
    edgefile0 = parse_edgefile(files[0])
    before = len(edgefile0.doy)
    filtdata = filter_edgedata(edgefile0, severity=severity)
    after = len(filtdata.doy)

    print(f"severity: {severity}  {before}->{after}")
