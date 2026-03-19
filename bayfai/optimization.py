import sys
import os
import numpy as np
import numpy.typing as npt
import h5py
import pyFAI
from pyFAI.geometry import Geometry
from pyFAI.goniometer import SingleGeometry
from pyFAI.geometryRefinement import GeometryRefinement
from pyFAI.calibrant import CALIBRANT_FACTORY
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
from mpi4py import MPI

sys.path.append("/sdf/home/l/lconreux/LCLSGeom")

from bayfai.geometry import azimuthal_integration, calculate_2theta

pyFAI.use_opencl = False

class BayFAIOpt:
    """
    Class to run BayFAI optimization on a powder image.

    Parameters
    ----------
    exp : str
        Experiment name
    run : int
        Run number
    """

    def __init__(
        self,
        exp,
        run,
    ):
        self.exp = exp
        self.run = run
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        if self.rank == 0:
            print(f"Getting {self.size} processes for BayFAIOpt task", flush=True)

    @staticmethod
    def UCB(X, gp_model, visited_idx, beta=1.96):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        ucb = y_pred + beta * y_std
        ucb[visited_idx] = -np.inf
        next = np.argmax(ucb)
        return next

    @staticmethod
    def q_UCB(X, gp_model, q, visited_idx, beta=1.96):
        y_pred, y_std = gp_model.predict(X, return_std=True)
        ucb = y_pred + beta * y_std
        ucb[visited_idx] = -np.inf
        top_next = np.argsort(ucb)[-q:]
        return top_next

    def setup(
        self,
        detname: str,
        powder: str,
        smooth: bool,
        calibrant: str,
        wavelength: float,
        fixed: list,
        in_file: str,
        is_psana2: bool = False,
    ):
        """
        Setup the BayFAI optimization.

        Parameters
        ----------
        detname : str
            Name of the detector
        powder : str
            Path to the powder image to use for calibration
        smooth : bool
            If True, apply smoothing to the powder image
        calibrant : PyFAI.Calibrant
            PyFAI calibrant object
        wavelength : float
            Wavelength of the X-ray source
        fixed : list
            List of parameters to keep fixed during optimization
        in_file : str
            Path to the input geometry file
        is_psana2 : bool
            If True, use psana2 geometry conversion

        Returns
        -------
        Imin : float
            Minimum intensity value for identifying Bragg peaks
        """
        self.detector = self.build_detector(in_file, is_psana2)
        self.powder = self.generate_powder(powder, detname, smooth, is_psana2)
        self.stacked_powder = np.reshape(self.powder, self.detector.shape)
        pos_pix = self.powder[self.powder > 0]
        self.Imin = np.percentile(pos_pix, 95)
        self.calibrant = self.define_calibrant(calibrant, wavelength)
        self.set_search_space(fixed)

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
            try:
                powder = h5[f"Sums/{detname}_calib_max"][()]
            except KeyError:
                print(
                    f"Cannot find {detname} Max powder in {powder_path}, defaulting to {detname} Sum instead."
                )
                try:
                    powder = h5[f"Sums/{detname}_calib"][()]
                except KeyError:
                    print(
                        f"Cannot find {detname} Sum powder in {powder_path}. Exiting..."
                    )
                    raise
        return powder

    def preprocess_powder(
        self,
        powder: npt.NDArray[np.float64],
        mask: npt.NDArray[np.integer],
        smooth: bool = False,
    ) -> npt.NDArray[np.float64]:
        """
        Preprocess extracted powder for enhancing optimization

        Parameters
        ----------
        powder : npt.NDArray[np.float64]
            Powder image to use for calibration
        mask : npt.NDArray[np.integer]
            Pixel mask to apply to the powder image
        smooth : bool, optional
            If True, apply smoothing to the powder image.
        """
        powder[powder < 0] = 0
        if smooth:
            for p in range(powder.shape[0]):
                gradx = np.gradient(powder[p], axis=0)
                grady = np.gradient(powder[p], axis=1)
                powder[p] = np.sqrt(gradx**2 + grady**2)
        powder[mask == 0] = 0
        self.powder_binary = (np.where(powder > np.mean(powder) + np.std(powder), 1.0, 0.0)).astype(np.float64)
        return powder
    
    def assemble_image(
        self, powder: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """
        Assemble the powder image from modules to full detector shape.

        Parameters
        ----------
        powder : npt.NDArray[np.float64]
            Powder image to use for calibration
        """
        pixel_index_map = self.detector.pixel_index_map
        max_rows = np.max(pixel_index_map[..., 0]) + 1
        max_cols = np.max(pixel_index_map[..., 1]) + 1
        assembled_powder = np.zeros((max_rows, max_cols))
        for p in range(pixel_index_map.shape[0]):
            i = pixel_index_map[p, ..., 0]
            j = pixel_index_map[p, ..., 1]
            assembled_powder[i, j] = powder[p]
        return assembled_powder

    def generate_powder(
        self, powder_path: str, detname: str, smooth: bool = False, is_psana2: bool = False,
    ) -> npt.NDArray[np.float64]:
        """
        Generate a preprocessed powder image from smalldata reduction.

        Parameters
        ----------
        powder_path : str
            Path to the h5 file containing the powder data.
        detname : str
            Name of the detector
        smooth : bool, optional
            If True, apply smoothing to the powder image.
        """
        mask = self.detector.geo.get_pixel_mask(mbits=3)
        if not is_psana2:
            mask = np.squeeze(mask, axis=0)
        powder = self.extract_powder(powder_path, detname)
        powder = self.preprocess_powder(powder, mask, smooth)
        self.assembled_powder = self.assemble_image(powder)
        return powder

    def build_detector(self, in_file: str, is_psana2: bool) -> pyFAI.detectors.Detector:
        """
        Read the metrology data and build a pyFAI detector object.

        Parameters
        ----------
        in_file : str
            Path to the Geometry .data file
        is_psana2 : bool
            If True, use psana2 geometry conversion

        Returns
        -------
        pyFAI.Detector
            Configured pyFAI detector object
        """
        if is_psana2:
            from LCLSGeom.psana2.converter import PsanaToPyFAI
            psana_to_pyfai = PsanaToPyFAI(
                input=in_file,
            )
        else:
            from LCLSGeom.psana.converter import PsanaToPyFAI
            psana_to_pyfai = PsanaToPyFAI(
                in_file=in_file,
            )
        detector = psana_to_pyfai.detector
        return detector

    def update_geometry(self, out_file: str, is_psana2: bool) -> pyFAI.detectors.Detector:
        """
        Update the geometry and write a new .poni, .geom and .data file

        Parameters
        ----------
        out_file : str
            Path to the output file
        is_psana2 : bool
            If True, use psana2 geometry conversion
        """
        path = os.path.dirname(out_file)
        poni_file = os.path.join(path, f"r{self.run:0>4}.poni")
        self.gr.save(poni_file)
        if is_psana2:
            from LCLSGeom.psana2.converter import PyFAIToPsana, PyFAIToCrystFEL, PsanaToPyFAI
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
                input=out_file,
                rotate=False,
            )
        else:
            from LCLSGeom.psana.converter import PyFAIToPsana, PyFAIToCrystFEL, PsanaToPyFAI
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
                in_file=out_file,
                rotate=False,
            )
        detector = psana_to_pyfai.detector
        return detector

    def define_calibrant(self, calibrant_name: str, wavelength: float) -> pyFAI.calibrant.Calibrant:
        """
        Define calibrant for optimization with appropriate wavelength

        Parameters
        ----------
        calibrant_name : str
            Name of the calibrant
        wavelength: float
            X-ray beam wavelength
        """
        self.calibrant_name = calibrant_name
        calibrant = CALIBRANT_FACTORY(calibrant_name)
        calibrant.wavelength = wavelength
        return calibrant

    def set_search_space(self, fixed: list) -> None:
        """
        Define the search space for the free parameters.

        Parameters
        ----------
        fixed : list
            List of parameters to keep fixed during optimization
        """
        self.fixed = fixed
        self.space = []
        parallelized = ["dist"]
        self.order = ["dist", "poni1", "poni2", "rot1", "rot2", "rot3"]
        for p in self.order:
            if p not in fixed and p not in parallelized:
                self.space.append(p)

    def distribute_distances(self, center, res):
        """
        Distribute distances across MPI ranks.

        Parameters
        ----------
        center : dict
            Center values for each parameter
        res : float
            Resolution of the grid used to discretize the parameter search space

        Returns
        -------
        dist : float
            The distance assigned to this MPI rank
        """
        half = self.size // 2
        offsets = (np.arange(self.size) - half) * res["dist"]
        distances = center["dist"] + offsets
        distances = np.round(distances, 6)
        self.distances = distances
        dist = distances[self.rank]
        return dist

    def create_search_space(self, dist, center, bounds, res):
        """
        Discretize the search space for the free parameters.

        Parameters
        ----------
        dist : float
            Distance on this MPI rank
        center : dict
            Center values for each parameter
        bounds : dict
            Bounds for each parameter, format: {param: (lower, upper)}
        res : dict
            Resolution per parameter

        Returns
        -------
        X : np.ndarray
            Full 6D geometry space (cartesian product)
        X_norm : np.ndarray
            Normalized search space (between-1 and 1)
        """
        center["dist"] = dist
        full_params = {}
        search_params = {}
        for p in self.order:
            if p in self.space:
                low = center[p] + bounds[p][0]
                high = center[p] + bounds[p][1]
                if high < low:
                    low, high = high, low
                step = res[p]
                full_params[p] = np.arange(low, high + step, step)
                search_params[p] = full_params[p]
            else:
                full_params[p] = np.array([center[p]])

        X = np.array(np.meshgrid(*[full_params[p] for p in self.order])).T.reshape(
            -1, len(self.order)
        )
        X_search = np.array(
            np.meshgrid(*[search_params[p] for p in self.space])
        ).T.reshape(-1, len(self.space))
        self.mins = np.min(X_search, axis=0)
        self.maxs = np.max(X_search, axis=0)
        X_norm = 2 * (X_search - self.mins) / (self.maxs - self.mins) - 1
        return X, X_norm

    def sample_initial_points(self, X, X_norm, center, bounds, n_samples, prior):
        """
        Sample initial points from the search space.

        Parameters
        ----------
        X : np.ndarray
            Search space
        X_norm : np.ndarray
            Normalized search space
        center : dict
            Center values for each parameter
        bounds : dict
            Bounds for each parameter
        n_samples : int
            Number of samples to draw
        prior : bool
            Use prior information for sampling

        Returns
        -------
        np.ndarray
            Sampled points
        """
        if prior:
            means = [center[p] for p in self.space]
            cov = np.diag(
                [(np.abs((bounds[p][1] - bounds[p][0])) / 5) ** 2 for p in self.space]
            )
            X_free = np.random.multivariate_normal(means, cov, n_samples)
            X_free = np.clip(X_free, self.mins, self.maxs)
            X_norm_samples = 2 * (X_free - self.mins) / (self.maxs - self.mins) - 1
            X_samples = np.tile([center[p] for p in self.order], (n_samples, 1))
            for i, p in enumerate(self.space):
                j = self.order.index(p)
                X_samples[:, j] = X_free[:, i]
            return X_samples, X_norm_samples
        else:
            idx_samples = np.random.choice(X.shape[0], n_samples)
            X_samples = X[idx_samples]
            X_norm_samples = X_norm[idx_samples]
            return X_samples, X_norm_samples

    def number_bragg_peaks(self, sample, Imin, max_rings):
        """
        Evaluate score at a given sampled geometry based on number of Bragg peaks found.

        Parameters
        ----------
        sample : list
            Geometry parameters
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider

        Returns
        -------
        score : float
            Scalar score for Bayesian optimization
        """
        dist, poni1, poni2, rot1, rot2, rot3 = sample
        geom_sample = Geometry(
            dist=dist,
            poni1=poni1,
            poni2=poni2,
            rot1=rot1,
            rot2=rot2,
            rot3=rot3,
            detector=self.detector,
            wavelength=self.calibrant.wavelength,
        )
        sg = SingleGeometry(
            "Score Geometry",
            self.stacked_powder,
            calibrant=self.calibrant,
            detector=self.detector,
            geometry=geom_sample,
        )
        sg.extract_cp(max_rings=max_rings, pts_per_deg=1, Imin=Imin)
        score = len(sg.geometry_refinement.data)
        return score

    def theta_residual(self, sample, Imin, max_rings):
        """
        Evaluate score at a given sampled geometry based on the residual between predicted and observed Bragg peak positions.

        Parameters
        ----------
        sample : list
            Geometry parameters
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider

        Returns
        -------
        score : float
            Scalar score for Bayesian optimization
        """
        dist, poni1, poni2, rot1, rot2, rot3 = sample
        geom_sample = Geometry(
            dist=dist,
            poni1=poni1,
            poni2=poni2,
            rot1=rot1,
            rot2=rot2,
            rot3=rot3,
            detector=self.detector,
            wavelength=self.calibrant.wavelength,
        )
        sg = SingleGeometry(
            "Score Geometry",
            self.stacked_powder,
            calibrant=self.calibrant,
            detector=self.detector,
            geometry=geom_sample,
        )
        sg.extract_cp(max_rings=max_rings, pts_per_deg=1, Imin=Imin)
        data = sg.geometry_refinement.data

        if data is None or len(data) == 0:
            return 0.0

        ix = data[:, 0]
        iy = data[:, 1]
        ring = data[:, 2].astype(np.int32)
        score = -np.log(sg.geometry_refinement.residu2(sample, ix, iy, ring) / len(data))
        return score

    def q_residual(self, sample, Imin, max_rings):
        """
        Evaluate score at a given sampled geometry based on the q-peaks
        found in the azimuthal integration.

        Parameters
        ----------
        sample: list
            Geometry Parameters
        Imin: float
            Minimum intensity threshold
        max_rings: int
            Maximum number of rings to consider
        """
        profile, ttha = azimuthal_integration(self.powder, self.detector, sample)
        ttha_min = np.min(ttha)
        ttha_max = np.max(ttha)

        expected_rings = np.array(self.calibrant.get_2th())
        valid_rings = (expected_rings >= ttha_min) & (expected_rings <= ttha_max)

        min_ring_delta = np.min(np.diff(expected_rings))
        min_resolution = np.min(np.diff(ttha))
        min_delta = min_ring_delta / min_resolution

        observed_rings = np.zeros_like(expected_rings)
        mean = np.mean(profile)
        std = np.std(profile)
        height = max(Imin, mean)
        prominence = std
        peaks, _ = find_peaks(profile, distance=min_delta, height=height, prominence=prominence)
        detected_rings = ttha[peaks]

        num_rings = min(len(peaks), max_rings)
        if num_rings == 0:
            return np.sum((observed_rings - expected_rings) ** 2) / len(expected_rings)

        ring_count = 0
        for i, is_valid in enumerate(valid_rings):
            if not is_valid:
                continue
            if ring_count >= num_rings:
                break
            observed_rings[i] = detected_rings[ring_count]
            ring_count += 1

        res = np.sum((observed_rings - expected_rings) ** 2) / len(expected_rings)
        score = -np.log(res)
        return score

    def q_residual_v2(self, sample, Imin, max_rings):
        """
        Evaluate score at a given sampled geometry based on the q-peaks
        found in the azimuthal integration.

        Parameters
        ----------
        sample: list
            Geometry Parameters
        Imin: float
            Minimum intensity threshold
        max_rings: int
            Maximum number of rings to consider
        """
        profile, ttha = azimuthal_integration(self.powder, self.detector, sample)
        ttha_min = np.min(ttha)
        ttha_max = np.max(ttha)

        tth = np.array(self.calibrant.get_2th())
        tth_min = np.zeros_like(tth)
        tth_max = np.zeros_like(tth)
        delta = (tth[1:] - tth[:-1]) / 10.0
        tth_max[:-1] = delta
        tth_max[-1] = delta[-1]
        tth_min[1:] = -delta
        tth_min[0] = -delta[0]
        tth_max += tth
        tth_min += tth
        valid_rings = (tth >= ttha_min) & (tth <= ttha_max)
        expected_rings = tth[valid_rings][:max_rings]
        lower = tth_min[valid_rings][:max_rings]
        upper = tth_max[valid_rings][:max_rings]

        if len(expected_rings) == 0:
            return 1

        res = 0.0
        for i, expected_ring in enumerate(expected_rings):
            mask = (ttha >= lower[i]) & (ttha <= upper[i])
            mean = np.mean(profile[mask])
            std = np.std(profile[mask])
            height = max(Imin, mean)
            peaks, _ = find_peaks(profile[mask], height=height, prominence=std)
            if len(peaks) == 0:
                res += expected_ring**2
            else:
                observed_ring = ttha[mask][peaks]
                expected_ring = np.tile(expected_ring, (len(observed_ring),))
                res += np.sum((observed_ring - expected_ring) ** 2)
        res /= len(expected_rings)
        score = -np.log(res)
        return score

    def powder_residual(self, sample):
        """
        Evaluate the fitting quality of the refined parameters by comparing
        the expected and observed ring positions.

        Parameters
        ----------
        sample : list
            Refined geometry parameters

        Returns
        -------
        fit_score : float
            Fitting quality score
        """
        theta = calculate_2theta(self.detector, sample)

        fit = np.zeros_like(theta)
        tth = np.array(self.calibrant.get_2th())
        upper = np.zeros_like(tth)
        lower = np.zeros_like(tth)
        delta = (tth[1:] - tth[:-1]) / 10.0
        upper[:-1] = delta
        upper[-1] = delta[-1]
        lower[1:] = -delta
        lower[0] = -delta[0]
        upper += tth
        lower += tth

        for i in range(len(tth)):
            mask = (theta >= lower[i]) & (theta <= upper[i])
            fit[mask] = 1.0
        fit_score = np.mean((fit - self.powder_binary) ** 2)
        score = -np.log(fit_score)
        return score

    def ring_intensity(self, sample, Imin, max_rings):
        """
        Evaluate score at a given sampled geometry based on the mean intensity
        of the found peaks in the azimuthal integration.

        Parameters
        ----------
        sample: list
            Geometry Parameters
        Imin: float
            Minimum intensity threshold
        max_rings: int
            Maximum number of rings to consider
        """
        profile, ttha = azimuthal_integration(self.powder, self.detector, sample)
        ttha_min = np.min(ttha)
        ttha_max = np.max(ttha)

        tth = np.array(self.calibrant.get_2th())
        tth_min = np.zeros_like(tth)
        tth_max = np.zeros_like(tth)
        delta = (tth[1:] - tth[:-1]) / 10.0
        tth_max[:-1] = delta
        tth_max[-1] = delta[-1]
        tth_min[1:] = -delta
        tth_min[0] = -delta[0]
        tth_max += tth
        tth_min += tth
        valid_rings = (tth >= ttha_min) & (tth <= ttha_max)
        expected_rings = tth[valid_rings][:max_rings]
        lower = tth_min[valid_rings][:max_rings]
        upper = tth_max[valid_rings][:max_rings]

        if len(expected_rings) == 0:
            return 0.0

        score = 0.0
        for i in range(len(expected_rings)):
            mask = (ttha >= lower[i]) & (ttha <= upper[i])
            mean = np.mean(profile[mask])
            std = np.std(profile[mask])
            height = max(Imin, mean)
            peaks, _ = find_peaks(profile[mask], height=height, prominence=std)
            if len(peaks) == 0:
                continue
            else:
                score += np.min(profile[mask][peaks])

        score /= len(expected_rings)
        return score

    def estimate_uncertainty(self, refinement, rel_eps=1e-3, abs_eps=1e-4):
        """
        Estimate parameter uncertainties from the Hessian matrix.

        Parameters
        ----------
        refinement : GeometryRefinement
            pyFAI refinement object after refine3.
        rel_eps : float
            Relative step for finite differences.
        abs_eps : float
            Absolute step for finite differences.

        Returns
        -------
        sigmas : np.ndarray
            Estimated uncertainties for each parameter
        is_min : bool
            True if a local minimum was found
        """
        param0 = np.array(
            [
                refinement.dist,
                refinement.poni1,
                refinement.poni2,
                refinement.rot1,
                refinement.rot2,
                refinement.rot3,
            ],
            dtype=np.float64,
        )
        param_names = ["dist", "poni1", "poni2", "rot1", "rot2"]
        size = len(param_names)

        d1 = refinement.data[:, 0]
        d2 = refinement.data[:, 1]
        ring = refinement.data[:, 2].astype(np.int32)
        f_min = refinement.residu2(param0, d1, d2, ring)
        hessian = np.zeros((size, size), dtype=np.float64)
        dof = max(len(refinement.data) - size, 1)

        delta = np.maximum(rel_eps * np.abs(param0), abs_eps)
        for i in range(size):
            deltai = delta[i]
            param = param0.copy()
            param[i] += deltai
            f_plus = refinement.residu2(param, d1, d2, ring)
            param = param0.copy()
            param[i] -= deltai
            f_minus = refinement.residu2(param, d1, d2, ring)
            hessian[i, i] = (f_plus + f_minus - 2.0 * f_min) / (deltai**2)

            for j in range(i + 1, size):
                deltaj = delta[j]
                param = param0.copy()
                param[i] += deltai
                param[j] += deltaj
                f_pp = refinement.residu2(param, d1, d2, ring)
                param = param0.copy()
                param[i] -= deltai
                param[j] -= deltaj
                f_mm = refinement.residu2(param, d1, d2, ring)
                param = param0.copy()
                param[i] += deltai
                param[j] -= deltaj
                f_pm = refinement.residu2(param, d1, d2, ring)
                param = param0.copy()
                param[i] -= deltai
                param[j] += deltaj
                f_mp = refinement.residu2(param, d1, d2, ring)
                hessian[j, i] = hessian[i, j] = (f_pp + f_mm - f_pm - f_mp) / (
                    4.0 * deltai * deltaj
                )

        eigs, _ = np.linalg.eigh(hessian)
        if np.any(eigs <= 0):
            sigmas = [np.inf] * size
            is_min = False
            penalty = 0.0
            return sigmas, is_min, penalty

        cov = np.linalg.inv(hessian)
        sigmas = f_min * np.diag(cov) / dof
        sigmas = np.sqrt(sigmas)
        is_min = True
        penalty = -np.log(np.linalg.det(cov)) / 2 
        return sigmas, is_min, penalty

    def gradient_descent(self, best_param, resolutions, Imin, max_rings, step=5):
        """
        Evaluate geometry found by BO on pyFAI refinement tool

        Parameters
        ----------
        best_param : list
            Best parameters found by Bayesian optimization
        resolutions : dict
            Resolution per parameter for restricted refinement
        Imin : float
            Minimum intensity threshold
        max_rings : int
            Maximum number of rings to consider
        step : int
            Size of the refinement space around best parameters

        Returns
        -------
        score : float
            Negative log of the residual after refinement
        sigma : np.ndarray
            Estimated uncertainties for each parameter
        penalty : float
            Penalty from uncertainty estimation
        size : int
            Number of Bragg peaks used in refinement
        params : dict
            Refined parameters
        is_min : bool
            Flag indicating if a local minimum was found
        """
        dist, poni1, poni2, rot1, rot2, rot3 = best_param
        best_geom = Geometry(
            dist=dist,
            poni1=poni1,
            poni2=poni2,
            rot1=rot1,
            rot2=rot2,
            rot3=rot3,
            detector=self.detector,
            wavelength=self.calibrant.wavelength,
        )
        sg = SingleGeometry(
            "Best Geometry",
            self.stacked_powder,
            calibrant=self.calibrant,
            detector=self.detector,
            geometry=best_geom,
        )
        sg.extract_cp(max_rings=max_rings, pts_per_deg=1, Imin=Imin)
        self.sg = sg

        if sg.geometry_refinement.data is None or len(sg.geometry_refinement.data) == 0:
            score = 0.0
            sigma = [np.inf] * 5
            penalty = 0.0
            size = 0
            is_min = False
            return score, sigma, penalty, size, best_param, is_min

        sg.geometry_refinement.set_dist_min(dist - step * resolutions["dist"])
        sg.geometry_refinement.set_dist_max(dist + step * resolutions["dist"])
        sg.geometry_refinement.set_poni1_min(poni1 - step * resolutions["poni1"])
        sg.geometry_refinement.set_poni1_max(poni1 + step * resolutions["poni1"])
        sg.geometry_refinement.set_poni2_min(poni2 - step * resolutions["poni2"])
        sg.geometry_refinement.set_poni2_max(poni2 + step * resolutions["poni2"])
        sg.geometry_refinement.set_rot1_min(rot1 - step * resolutions["rot1"])
        sg.geometry_refinement.set_rot1_max(rot1 + step * resolutions["rot1"])
        sg.geometry_refinement.set_rot2_min(rot2 - step * resolutions["rot2"])
        sg.geometry_refinement.set_rot2_max(rot2 + step * resolutions["rot2"])
        fix = ["rot3", "wavelength"]
        score = -np.log(sg.geometry_refinement.refine3(fix=fix))
        sigma, is_min, penalty = self.estimate_uncertainty(sg.geometry_refinement)
        params = sg.geometry_refinement.param
        size = len(sg.geometry_refinement.data)
        return score, sigma, penalty, size, params, is_min

    @ignore_warnings(category=ConvergenceWarning)
    def bayes_opt_distance(
        self,
        dist,
        center,
        bounds,
        res,
        score,
        n_samples,
        n_iterations,
        Imin,
        max_rings,
        beta=1.96,
        step=5,
        prior=True,
        seed=None,
    ):
        """
        Run Bayesian Optimization on a subspace of fixed distance.

        Parameters
        ----------
        dist : float
            Distance on this MPI rank
        center : dict
            Dictionary of center values for each parameter
        bounds : dict
            Dictionary of bounds for each parameter
        res : dict
            Dictionary of resolution for each parameter
        score : str
            Scoring method to use: 'bragg', 'residual', 'residual_v2', 'theta_residual', 'intensity'
        n_samples : int
            Number of samples to initialize the Gaussian Process
        n_iterations : int
            Number of iterations of Bayesian Optimization
        Imin : float
            Minimum intensity threshold for identifying Bragg peaks
        max_rings : int
            Maximum number of rings to search for Bragg peaks
        beta : float
            Exploration-exploitation trade-off parameter for UCB acquisition function
        step : int
            Size of the refinement space around best parameters
        prior : bool
            Whether to sample initial points around the center or randomly
        seed : optional, int
            Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)

        # 1. Create the search space
        X, X_norm = self.create_search_space(dist, center, bounds, res)

        # 2. Sample initial points
        X_samples, X_norm_samples = self.sample_initial_points(
            X, X_norm, center, bounds, n_samples, prior
        )

        # 3. Evaluate the initial points
        bo_history = {"params": [], "scores": []}
        y = np.zeros((n_samples))
        for i in range(n_samples):
            if score == "bragg":
                y[i] = self.number_bragg_peaks(X_samples[i], Imin, max_rings)
            elif score == "residual":
                y[i] = self.q_residual(X_samples[i], Imin, max_rings)
            elif score == "residual_v2":
                y[i] = self.q_residual_v2(X_samples[i], Imin, max_rings)
            elif score == "theta_residual":
                y[i] = self.theta_residual(X_samples[i], Imin, max_rings)
            elif score == "intensity":
                y[i] = self.ring_intensity(X_samples[i], Imin, max_rings)
            bo_history["params"].append(X_samples[i])
            bo_history["scores"].append(y[i])

        if np.all(y == 0.0):
            result = {
                "bo_history": bo_history,
                "params": [dist, 0, 0, 0, 0, 0],
                "score": 0.0,
                "sigma": [np.inf] * 5,
                "penalty": 0.0,
                "size": 0,
                "best_idx": 0,
                "is_min": False,
            }
            print(
                f"All samples have score 0 for dist={dist}. Skipping Bayesian Optimization.", flush=True,
            )
            return result

        y[np.isnan(y)] = 0
        if np.std(y) != 0:
            y_norm = (y - np.mean(y)) / np.std(y)
        else:
            y_norm = y - np.mean(y)

        # 4. Initialize the Gaussian Process model
        kernel = RBF(length_scale=0.3, length_scale_bounds=(0.2, 0.4)) * ConstantKernel(
            constant_value=1.0, constant_value_bounds=(0.5, 1.5)
        ) + WhiteKernel(noise_level=0.001, noise_level_bounds="fixed")
        gp_model = GaussianProcessRegressor(
            kernel=kernel, n_restarts_optimizer=10, random_state=0
        )
        gp_model.fit(X_norm_samples, y_norm)
        visited_idx = list([])

        # 5. Run the Bayesian Optimization loop
        for i in range(n_iterations):
            # 6. Select the next point to evaluate
            next = self.UCB(X_norm, gp_model, visited_idx, beta)
            next_sample = X[next]
            visited_idx.append(next)

            # 7. Compute the score of the next point
            if score == "bragg":
                yi = self.number_bragg_peaks(X_samples[i], Imin, max_rings)
            elif score == "residual":
                yi = self.q_residual(X_samples[i], Imin, max_rings)
            elif score == "residual_v2":
                yi = self.q_residual_v2(X_samples[i], Imin, max_rings)
            elif score == "theta_residual":
                yi = self.theta_residual(X_samples[i], Imin, max_rings)
            elif score == "intensity":
                yi = self.ring_intensity(X_samples[i], Imin, max_rings)
            y = np.append(y, [yi], axis=0)
            bo_history["params"].append(next_sample)
            bo_history["scores"].append(yi)
            X_samples = np.append(X_samples, [X[next]], axis=0)
            X_norm_samples = np.append(X_norm_samples, [X_norm[next]], axis=0)
            if np.std(y) != 0:
                y_norm = (y - np.mean(y)) / np.std(y)
            else:
                y_norm = y - np.mean(y)

            # 8. Update the Gaussian Process model
            gp_model.fit(X_norm_samples, y_norm)

        # 9. Gather results
        best_idx = np.argmax(y)
        best_param = X_samples[best_idx]
        score, sigma, penalty, size, params, is_min = self.gradient_descent(
            best_param, res, Imin, max_rings, step
        )
        print(
            f"Rank {self.rank} dist={dist:.4f}m: score={score}", flush=True,
        )
        result = {
            "bo_history": bo_history,
            "params": params,
            "score": score,
            "sigma": sigma,
            "penalty": penalty,
            "size": size,
            "best_idx": best_idx,
            "is_min": is_min,
        }
        return result

    def bayfai_opt(
        self,
        center,
        bounds,
        res,
        n_samples,
        n_iterations,
        max_rings,
        beta=1.96,
        step=5,
        prior=True,
        seed=None,
        score="bragg",
    ):
        """
        Run BayFAI optimization.
        Split the distance parameter across MPI ranks.
        Run Bayesian Optimization on each rank with fixed distance.
        Perform pyFAI least-squares refinement for each rank's best geometry.
        Optimal geometry is chosen based on the lowest residual among ranks.

        Parameters
        ----------
        center : dict
            Dictionary of center values for each parameter
        bounds : dict
            Dictionary of bounds for each parameter
        res : dict
            Dictionary of resolution for each parameter
        n_samples : int
            Number of samples to initialize the Gaussian Process
        n_iterations : int
            Number of iterations of Bayesian Optimization
        max_rings : int
            Maximum number of rings to consider
        beta : float
            Exploration-exploitation trade-off parameter for UCB acquisition function
        prior : bool
            Whether to sample initial points around the center or randomly
        seed : int
            Random seed for reproducibility
        score : str
            Score function to use
        """
        dist = self.distribute_distances(center, res)
        print(
            f"Rank {self.rank}: Running Bayesian Optimization on distance {dist:.4f} m", flush=True,
        )

        bayfai_hyperparams = {
            "n_samples": n_samples,
            "n_iterations": n_iterations,
            "Imin": self.Imin,
            "max_rings": max_rings,
            "beta": beta,
            "step": step,
            "prior": prior,
            "seed": seed,
        }

        results = self.bayes_opt_distance(
            dist,
            center,
            bounds,
            res,
            score,
            **bayfai_hyperparams,
        )

        self.comm.Barrier()

        self.scan = {}
        self.scan["bo_history"] = self.comm.gather(results["bo_history"], root=0)
        self.scan["params"] = self.comm.gather(results["params"], root=0)
        self.scan["score"] = self.comm.gather(results["score"], root=0)
        self.scan["size"] = self.comm.gather(results["size"], root=0)
        self.scan["sigma"] = self.comm.gather(results["sigma"], root=0)
        self.scan["penalty"] = self.comm.gather(results["penalty"], root=0)
        self.scan["best_idx"] = self.comm.gather(results["best_idx"], root=0)
        self.scan["is_min"] = self.comm.gather(results["is_min"], root=0)
        self.finalize()

    def finalize(self, lbda=0.2):
        if self.rank == 0:
            for key in self.scan.keys():
                self.scan[key] = np.array([item for item in self.scan[key]])
            self.valid = self.scan["is_min"]
            self.invalid = np.where(~self.valid)[0]
            self.final_score = self.scan["score"] + lbda * self.scan["penalty"]
            self.index = np.argmax(self.final_score)
            self.bo_history = self.scan["bo_history"][self.index]
            self.params = self.scan["params"][self.index]
            self.neglog_score = self.scan["score"][self.index]
            self.size = self.scan["size"][self.index]
            self.sigma = self.scan["sigma"][self.index]
            self.penalty = self.scan["penalty"][self.index]
            self.best_idx = self.scan["best_idx"][self.index]
            self.gr = GeometryRefinement(
                calibrant=self.calibrant,
                dist=self.params[0],
                poni1=self.params[1],
                poni2=self.params[2],
                rot1=self.params[3],
                rot2=self.params[4],
                rot3=self.params[5],
                detector=self.detector,
                wavelength=self.calibrant.wavelength,
            )

    def grid_search_distance(
        self,
        dist,
        center,
        bounds,
        res,
        max_rings,
        score,
        out_dir,
    ):
        """
        Run Grid Search on a fixed distance.

        Parameters
        ----------
        dist : float
            Distanc
        """
        # 1. Create the search space
        X, _ = self.create_search_space(dist, center, bounds, res)

        # 2. Evaluate Points
        history = {"params": [], "scores": []}
        Imin = np.percentile(self.powder, 95)
        y = np.zeros((X.shape[0]))
        for i in range(X.shape[0]):
            if score == "bragg":
                y[i] = self.number_bragg_peaks(X[i], Imin, max_rings)
            elif score == "residual":
                y[i] = self.q_residual(X[i], Imin, max_rings)
            elif score == "residual_v2":
                y[i] = self.q_residual_v2(X[i], Imin, max_rings)
            elif score == "theta_residual":
                y[i] = self.theta_residual(X[i], Imin, max_rings)
            elif score == "intensity":
                y[i] = self.ring_intensity(X[i], Imin, max_rings)
            elif score == "powder_residual":
                y[i] = self.powder_residual(X[i])
            history["params"].append(X[i])
            history["scores"].append(y[i])
        
        # 3. Save results
        filename = f"{self.exp}_r{self.run:04d}_score_{score}_dist_{str(dist).replace('.', '')}"
        np.save(f"{out_dir}/{filename}.npy", history["scores"])
        self.params = history["params"]

        return history

    def grid_search(
        self,
        center,
        bounds,
        res,
        max_rings,
        score,
        out_dir,
    ):
        """
        Run Grid Search on all parameter space
        Split the distance parameter across MPI ranks.
        Run Grid Search on each rank with fixed distance.

        Parameters
        ----------
        center : dict
            Dictionary of center values for each parameter
        bounds : dict
            Dictionary of bounds for each parameter
        res : dict
            Dictionary of resolution for each parameter
        max_rings : int
            Maximum number of rings to consider
        score : str
            Score function to use
        out_dir : str
            Output Directory path
        """
        dist = self.distribute_distances(center, res)
        print(
            f"Rank {self.rank}: Running Bayesian Optimization on distance {dist:.4f} m", flush=True,
        )

        history = self.grid_search_distance(
            dist,
            center,
            bounds,
            res,
            max_rings,
            score,
            out_dir,
        )

        self.comm.Barrier()

        self.scan = self.comm.gather(history, root=0)

        if self.rank == 0:
            best_dist = 0
            best_score = 0
            best_idx = 0 
            for i in range(len(self.scan)):
                scores = self.scan[i]["scores"]
                j = np.argmax(scores)
                if scores[j] > best_score:
                    best_dist = i
                    best_score = scores[j]
                    best_idx = j
            self.best_dist = best_dist
            self.best_score = best_score
            self.best_index = best_idx
            filename = f"{self.exp}_r{self.run:04d}_params"
            np.save(f"{out_dir}/{filename}.npy", self.scan[i]["params"])
            print("Grid Search finished!")
