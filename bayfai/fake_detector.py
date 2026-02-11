import os
import sys
sys.path.append("/sdf/home/l/lconreux/LCLSGeom")

import psana

if hasattr(psana, "xtc_version"):
    from psana import DataSource
    from LCLSGeom.psana2.converter import PsanaToPyFAI, PyFAIToPsana, PyFAIToCrystFEL

    IS_PSANA2 = True
else:
    from psana import DataSource, Detector
    from LCLSGeom.psana.converter import PsanaToPyFAI, PyFAIToPsana, PyFAIToCrystFEL

    IS_PSANA2 = False

import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib import lines
import ipywidgets as widgets
from IPython.display import display
import numpy.typing as npt
from typing import Optional
import pyFAI
from pyFAI.calibrant import CALIBRANT_FACTORY
from pyFAI.goniometer import SingleGeometry
from pyFAI.azimuthalIntegrator import AzimuthalIntegrator
from pyFAI.geometry import Geometry
from pyFAI.units import RADIAL_UNITS

from bayfai.geometry import calculate_2theta

class FakeDetector:
    """
    Class for manually solving geometry using PyFAI

    Parameters
    ----------
    exp : str
        Experiment tag
    run : int
        Run number
    detname : str
        Detector name (epix10k2M, jungfrau4M, Rayonix, Epix10kaQuad.1...)
    calibrant : str
        Calibrant name (AgBh, LaB6, CeO2)
    powder_path : str
        Path to the h5 file containing the powder data.
    in_file : Optional[str]
        Path to the .data file containing the detector information.
    """
    def __init__(self, exp, run, detname, calibrant, powder_path, in_file=None):
        self.exp = exp
        self.run = run
        self.detname = detname
        self.calibrant_name = calibrant
        self.params = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.min_rings = 1
        self.max_rings = 10 
        if IS_PSANA2:
            self.ds = DataSource(exp=exp, run=run, skip_calib_load="all", max_events=1)
            self.runs = next(self.ds.runs())
            self.evt = next(self.runs.events())
        else:
            self.ds = DataSource(f"exp={exp}:run={run}:idx")
            self.runs = next(self.ds.runs())
            self.evt = self.runs.event(self.runs.times()[0])
        self.setup(detname, powder_path, calibrant, in_file)

    def setup(
        self,
        detname: str,
        powder_path: str,
        calibrant: str,
        in_file: Optional[str] = None,
    ):
        """
        Setup the BayFAI optimization.

        Parameters
        ----------
        detname : str
            Name of the detector
        powder_path : str
            Path to the powder image to use for calibration
        calibrant : PyFAI.Calibrant
            PyFAI calibrant object
        in_file : str
            Path to the input geometry file
        """
        self.detector = self.build_detector(detname, in_file)
        self.calibrant = self.define_calibrant(calibrant)
        self.powder = self.generate_powder(powder_path, detname)

    def extract_powder(self, powder_path: str, detname: str) -> npt.NDArray[np.float64]:
        """
        Extract a powder image from smalldata analysis.

        Parameters
        ----------
        powder_path : str
            Path to the h5 file containing the powder data.

        Returns
        -------
        powder : npt.NDArray[np.float64]
            The extracted powder image.
        """
        with h5py.File(powder_path) as h5:
            if 'quad' in detname.lower():
                detname = detname.replace('.', '')
            try:
                powder = h5[f"Sums/{detname}_calib_max"][()]
            except KeyError:
                print(
                    f"Cannot find {detname} Max powder in {powder_path}, defaulting to {detname} Sum instead."
                )
                powder = h5[f"Sums/{detname}_calib"][()]
        return powder

    def preprocess_powder(self, powder: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Preprocess extracted powder for enhancing optimization

        Parameters
        ----------
        powder : npt.NDArray[np.float64]
            Powder image to use for calibration
        """
        powder[powder < 0] = 0
        index_max_x = np.max(self.detector.pixel_index_map[..., 0]) + 1
        index_max_y = np.max(self.detector.pixel_index_map[..., 1]) + 1
        powder_ass = np.zeros((index_max_x, index_max_y))
        mask_ass = np.zeros((index_max_x, index_max_y))
        for p in range(self.detector.n_modules):
            i = self.detector.pixel_index_map[p, ..., 0]
            j = self.detector.pixel_index_map[p, ..., 1]
            mask_ass[i, j] = self.mask[p]
            powder_ass[i, j] = powder[p]
        powder_ass *= mask_ass
        powder[self.mask == 0] = 0
        return powder_ass, powder

    def generate_powder(self, powder_path: str, detname: str):
        """
        Generate the assembled powder plot and cache it.

        Parameters
        ----------
        powder_path : str
            Path to the h5 file containing the powder data.
        detname : str
            Name of the detector.
        max_rings : int
            Maximum number of rings to display.
        """
        powder = self.extract_powder(powder_path, detname)
        powder_ass, powder = self.preprocess_powder(powder)
        self.unass_powder = powder
        self.stacked_powder = np.reshape(powder, self.detector.shape)
        self.vmin = np.percentile(powder, 5)
        self.vmax = np.percentile(powder, 95)
        self._cached_powder = powder_ass
        fig, ax = plt.subplots(figsize=(8, 8))
        img = ax.imshow(self._cached_powder, vmin=self.vmin, vmax=self.vmax, origin="lower")
        ttha = calculate_2theta(self.detector, self.params)
        contours = []
        for p in range(self.detector.raw_shape[0]):
            i = self.detector.pixel_index_map[p, ..., 0]
            j = self.detector.pixel_index_map[p, ..., 1]
            cs = ax.contour(j, i, ttha[p], levels=self.tth[:self.max_rings], cmap="autumn",
                            linewidths=1.2, linestyles="dashed")
            contours.append(cs)
        self._cached_fig = fig
        self._cached_img = img
        self._cached_ax = ax
        self._cached_contours = contours
        return powder_ass

    def show_powder(self):
        """
        Display powder image with contrast and geometry parameters to overlay diffraction rings.
        """
        fig = self._cached_fig
        ax = self._cached_ax
        img = self._cached_img
        powder = self._cached_powder

        vmin_slider = widgets.FloatSlider(value=5.0, min=0.0, max=100.0, step=0.1, description="vmin (%)")
        vmax_slider = widgets.FloatSlider(value=95.0, min=0.0, max=100.0, step=0.1, description="vmax (%)")
        min_rings_slider = widgets.IntSlider(value=self.min_rings, min=1, max=len(self.tth), step=1, description="First Ring to Show")
        max_rings_slider = widgets.IntSlider(value=self.max_rings, min=1, max=len(self.tth), step=1, description="Last Ring to Show")
        image_box = widgets.VBox([vmin_slider, vmax_slider, min_rings_slider, max_rings_slider])
        image_box = widgets.VBox([widgets.Label("Image Settings"), image_box])

        geom_labels = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        steps = [1e-3, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3]
        geom_inputs = [widgets.FloatText(value=p, description=label, step=step)
                    for p, label, step in zip(self.params, geom_labels, steps)]
        geom_box = widgets.VBox(geom_inputs)
        geom_box = widgets.VBox([widgets.Label("Geometry Parameters"), geom_box])

        def _update(change=None):
            if vmin_slider.value >= vmax_slider.value:
                return
            self.vmin = np.percentile(powder, vmin_slider.value)
            self.vmax = np.percentile(powder, vmax_slider.value)
            img.set_clim(self.vmin, self.vmax)

            if min_rings_slider.value >= max_rings_slider.value:
                return
            self.min_rings = min_rings_slider.value
            self.max_rings = max_rings_slider.value

            for cs in self._cached_contours:
                for c in cs.collections:
                    c.remove()

            self.params = [w.value for w in geom_inputs]
            ttha = calculate_2theta(self.detector, self.params)
            contours = []
            for p in range(self.detector.raw_shape[0]):
                i = self.detector.pixel_index_map[p, ..., 0]
                j = self.detector.pixel_index_map[p, ..., 1]
                cs = ax.contour(j, i, ttha[p], levels=self.tth[self.min_rings-1:self.max_rings], cmap="autumn",
                                linewidths=1.2, linestyles="dashed")
                contours.append(cs)
            self._cached_contours = contours
            fig.canvas.draw_idle()

        for w in [vmin_slider, vmax_slider, min_rings_slider, max_rings_slider, *geom_inputs]:
            w.observe(_update, names="value")

        controls = widgets.VBox([image_box, geom_box])
        ui = widgets.HBox([fig.canvas, controls])
        display(ui)

    def build_detector(self, detname: str, in_file: Optional[str] = None) -> pyFAI.detectors.Detector:
        """
        Build a PyFAI detector from a .data file

        Parameters
        ----------
        detname : str
            Name of the detector
        in_file : str
            Path to the .data file containing the detector information.
        
        Returns
        -------
        detector : pyFAI.detectors.Detector
            The built PyFAI detector.
        """
        if IS_PSANA2:
            det = self.runs.Detector(detname)
            psana_to_pyfai = PsanaToPyFAI(
                input=det,
            )
            detector = psana_to_pyfai.detector
            self.mask = detector.geo.get_pixel_mask(mbits=3)
        else:
            psana_to_pyfai = PsanaToPyFAI(
                in_file=in_file,
            )
            detector = psana_to_pyfai.detector
            mask = detector.geo.get_pixel_mask(mbits=3)
            self.mask = np.squeeze(mask, axis=0)
        return detector

    def define_calibrant(self, calibrant_name: str) -> pyFAI.calibrant.Calibrant:
        """
        Define calibrant for optimization with appropriate wavelength

        Parameters
        ----------
        calibrant_name : str
            Name of the calibrant
        exp : str
            Name of the experiment
        run : int
            Run number
        """
        self.calibrant_name = calibrant_name
        calibrant = CALIBRANT_FACTORY(calibrant_name)
        if IS_PSANA2:
            try:
                det_photon_energy = self.runs.Detector("ebeamh")
                photon_energy = det_photon_energy.raw.ebeamPhotonEnergy(self.evt)
                wavelength = 1.23984197386209e-06 / photon_energy
            except Exception:
                det_wavelength = self.runs.Detector("SIOC:SYS0:ML00:AO192")
                wavelength = det_wavelength(self.evt) * 1e-9
        else:
            try:
                det_photon_energy = Detector("EBeam")
                photon_energy = det_photon_energy.get(self.evt).ebeamPhotonEnergy()
                wavelength = 1.23984197386209e-06 / photon_energy
            except Exception:
                wavelength = self.ds.env().epicsStore().value("SIOC:SYS0:ML00:AO192") * 1e-9
        calibrant.wavelength = wavelength
        self.tth = np.array(calibrant.get_2th())
        return calibrant

    def azimuthal_integration(self) -> tuple:
        """
        Compute the radial intensity profile of an image.

        Parameters
        ----------
        powder : numpy.ndarray, shape (n,m)
            detector image
        detector : pyFAI.Detector, shape (n,m)
            PyFAI detector object
        params : list, optional
            6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
        """
        if self.params is not None:
            ai = AzimuthalIntegrator(
                detector=self.detector,
                dist=self.params[0],
                poni1=self.params[1],
                poni2=self.params[2],
                rot1=self.params[3],
                rot2=self.params[4],
                rot3=self.params[5],
                wavelength=self.calibrant.wavelength,
            )
        else:
            ai = AzimuthalIntegrator(detector=self.detector,
                dist=0.1,
                wavelength=self.calibrant.wavelength
            )
        q, I = ai.integrate1d(
            self.stacked_powder,
            npt=256,
            unit="q_A^-1",
            method="cython")
        return q, I

    def integrate_detector(self):
        """
        Integrate azimuthally powder diffraction rings based on the geometry and overlay expected diffraction rings.

        Parameters
        ----------
        params:
            6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
        """
        q, I = self.azimuthal_integration()

        fig, ax = plt.subplots(figsize=(10, 4)) 
        unit = RADIAL_UNITS["q_A^-1"]
        ax.plot(q, I, color="black", linewidth=0.8)
        x_values = self.calibrant.get_peaks(unit)
        if x_values is not None:
            for x in x_values:
                line = lines.Line2D(
                    [x, x],
                    ax.axis()[2:4],
                    color="red",
                    linestyle="--",
                    linewidth=0.8,
                    alpha=0.7,
                )
                ax.add_line(line)

        ax.set_title("Radial Profile")
        if unit:
            ax.set_xlabel(unit.label)
        ax.set_ylabel("Intensity")

    def refine_geometry(self):
        """
        Refine the geometry parameters based on the detector and current parameters.

        Returns
        -------
        refined_params : list
            Refined geometry parameters.
        """
        geom = Geometry(
            dist=self.params[0],
            poni1=self.params[1],
            poni2=self.params[2],
            rot1=self.params[3],
            rot2=self.params[4],
            rot3=self.params[5],
            wavelength=self.calibrant.wavelength,
            detector=self.detector,
        )
        sg = SingleGeometry(
            label="",
            image=self.stacked_powder,
            calibrant=self.calibrant,
            detector=self.detector,
            geometry=geom,
        )
        sg.extract_cp(max_rings=self.max_rings, pts_per_deg=1, Imin=self.vmin)
        sg.geometry_refinement.set_dist_min(self.params[0] - 0.005)
        sg.geometry_refinement.set_dist_max(self.params[0] + 0.005)
        sg.geometry_refinement.set_poni1_min(self.params[1] - 0.0005)
        sg.geometry_refinement.set_poni1_max(self.params[1] + 0.0005)
        sg.geometry_refinement.set_poni2_min(self.params[2] - 0.0005)
        sg.geometry_refinement.set_poni2_max(self.params[2] + 0.0005)
        sg.geometry_refinement.set_rot1_min(self.params[3] - 0.1)
        sg.geometry_refinement.set_rot1_max(self.params[3] + 0.1)
        sg.geometry_refinement.set_rot2_min(self.params[4] - 0.1)
        sg.geometry_refinement.set_rot2_max(self.params[4] + 0.1)
        fix = ["rot3", "wavelength"]
        residual = sg.geometry_refinement.refine3(fix=fix)
        self.params = sg.geometry_refinement.param
        print("Final Geometry Score:", residual)
        print("Distance (m):", self.params[0])
        print("X-shift (m):", self.params[1])
        print("Y-shift (m):", self.params[2])    
        print("Rx (rad):", self.params[3])
        print("Ry (rad):", self.params[4])
        print("Rz (rad):", self.params[5])
        self.gr = sg.geometry_refinement

    def update_geometry(self, out_file: str) -> pyFAI.detectors.Detector:
        """
        Update the geometry and write a new .poni, .geom and .data file

        Parameters
        ----------
        out_file : str
            Path to the output file
        """
        path = os.path.dirname(out_file)
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        poni_file = os.path.join(path, f"r{self.run:0>4}.poni")
        self.gr.save(poni_file)
        PyFAIToPsana(
            in_file=poni_file,
            detector=self.detector,
            out_file=out_file,
        )
        geom_file = os.path.join(path, f"r{self.run:0>4}.geom")
        PyFAIToCrystFEL(
            in_file=poni_file,
            detector=self.detector,
            out_file=geom_file,
        )
        psana_to_pyfai = PsanaToPyFAI(
            out_file,
        )
        detector = psana_to_pyfai.detector
        return detector