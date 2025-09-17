import pyarts
import numpy as np
from pathlib import Path
from .logger import get_logger
from .utils import find_downloads


class Ycalc:
    def __init__(self, start, end, nf, summer, save):
        self.logger = get_logger()
        self.start = start
        self.end = end
        self.nf = nf
        self.summer = summer
        self.lat = [67.8]
        self.lon = [20.22]
        self.f0 = 273.051010e9
        self.arts = pyarts.workspace.Workspace()

        self.set_catalogue()
        self.set_line()
        self.set_grids(summer=summer)
        self.set_radiative_agendas()
        self.set_sensor_and_geometrics()
        self.set_lines_per_species(save=save)
        self.checks()
        self.ycalc(save=save)

    def set_line(self):
        if self.start is None:
            self.start = 250e9
        elif self.end is None:
            self.end = 300e9

        if self.start > self.end:
            end = self.start
            self.start = self.end
            self.end = end

        self.arts.f_grid = np.linspace(self.start, self.end, self.nf)
        self.arts.abs_speciesSet(
            species=[
                f"O2-Z-*-{self.start - 1}-{self.end + 1}",
                f"O3-*-{self.start - 1}-{self.end + 1}",
                f"N2O-*-{self.start - 1}-{self.end + 1}",
                f"HNO3-*-{self.start - 1}-{self.end + 1}",
                "H2O-PWR98",
                f"H2O-*-{self.start - 1}-{self.end + 1}"
            ]
        )
        self.arts.Wigner6Init()

    def set_catalogue(self):
        home = Path.home()
        catalogue_path = home / ".cache/arts/"

        if not catalogue_path.exists():
            self.logger.warning(f"Catalogue data not found. Downloading into {catalogue_path}\n")
            pyarts.cat.download.retrieve(verbose=True)

        self.cat_data = catalogue_path / "arts-cat-data-2.6.16"
        self.xml_data = catalogue_path / "arts-xml-data-2.6.16"

    def set_grids(self, summer=False):
        self.arts.p_grid = np.logspace(np.log10(105000), np.log10(0.1))
        fascod = "planets/Earth/Fascod"

        if summer:
            basename = self.xml_data / fascod / "subarctic-summer/subarctic-summer"
        else:
            basename = self.xml_data / fascod / "subarctic-winter/subarctic-winter"

        self.arts.AtmRawRead(basename=str(basename))
        self.arts.lat_grid = np.linspace(50, 80)
        self.arts.lon_grid = np.linspace(-180, 180)
        self.arts.AtmosphereSet3D()
        self.arts.AtmFieldsCalcExpand1D()
        self.arts.refellipsoidEarth(model="Sphere")
        self.arts.MagFieldsCalcIGRF(time=pyarts.arts.Time("2025-09-15 12:00:00"))

    def set_radiative_agendas(self):
        self.arts.iy_main_agendaSet(option="Emission")
        self.arts.iy_surface_agendaSet(option="UseSurfaceRtprop")
        self.arts.surface_rtprop_agendaSet(option="Blackbody_SurfTFromt_field")
        self.arts.iy_space_agendaSet(option="CosmicBackground")
        self.arts.water_p_eq_agendaSet(option="MK05")
        self.arts.iy_unit = "PlanckBT"
        self.arts.stokes_dim = 4
        self.arts.ppath_lmax = 1e3
        self.arts.ppath_lraytrace = 1e3
        self.arts.rt_integration_option = "default"
        self.arts.nlteOff()

        self.arts.jacobianOff()
        self.arts.cloudboxOff()

    def set_sensor_and_geometrics(self):
        self.arts.z_surfaceConstantAltitude(altitude=0.0)

        self.arts.sensor_pos = [[411.0, self.lat[0], self.lon[0]]]
        self.arts.sensor_los = [[0, 0]]
        self.arts.ppath_agendaSet(option="FollowSensorLosPath")
        self.arts.ppath_step_agendaSet(option="GeometricPath")
        self.arts.sensorOff()

    def set_lines_per_species(self, save):
        self.logger.info(f"Calculating abs lines")
        self.arts.abs_lines_per_speciesReadSpeciesSplitCatalog(
            basename=f"{self.cat_data}/lines/"
        )
        self.arts.propmat_clearsky_agendaAuto()

    def checks(self):
        self.arts.lbl_checkedCalc()
        self.arts.atmgeom_checkedCalc()
        self.arts.atmfields_checkedCalc()
        self.arts.cloudbox_checkedCalc()
        self.arts.sensor_checkedCalc()
        self.arts.propmat_clearsky_agenda_checkedCalc()

    def ycalc(self, save):
        savedir = find_downloads()
        if save is None:
            savename = f"{int(self.start)}_{int(self.end)}.npy"
        else:
            savename = f"{save}.npy"

        savepath = savedir / savename
        self.logger.info("Starting yCalc")
        self.arts.yCalc()
        self.logger.info(f"yCalc done and saving to:\n{savepath}")
        sI = self.arts.y.value[0::4]
        sQ = self.arts.y.value[1::4]
        sU = self.arts.y.value[2::4]
        sV = self.arts.y.value[3::4]
        data = {"f": self.arts.f_grid.value,
                "I": sI,
                "Q": sQ,
                "U": sU,
                "V": sV}
        np.save(savepath, data)
