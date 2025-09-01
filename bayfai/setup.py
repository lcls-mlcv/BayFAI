import sys
import numpy as np
import h5py
from pyFAI.calibrant import CALIBRANT_FACTORY
from scipy.ndimage import gaussian_filter

sys.path.append("/sdf/home/l/lconreux/LCLSGeom")
from LCLSGeom.psana.converter import PsanaToPyFAI

def extract_powder(powder_path, detname):
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
            print('No "Max" powder found in SmallData. Using "Sum" powder.')
            powder = h5[f"Sums/{detname}_calib"][()]
    return powder

def preprocess_powder(powder, smooth=False):
    """
    Preprocess extracted powder for enhancing optimization

    Parameters
    ----------
    powder : npt.NDArray[np.float64]
        Powder image to use for calibration
    shape : tuple
        Stacked detector shape
    smooth : bool, optional
        If True, apply smoothing to the powder image.
    """
    powder[powder < 0] = 0
    if smooth:
        calib = gaussian_filter(powder, sigma=1)
        gradx_calib = np.zeros_like(powder)
        grady_calib = np.zeros_like(powder)
        for p in range(powder.shape[0]):
            gradx_calib[p, :-1, :-1] = (
                calib[p, 1:, :-1] - calib[p, :-1, :-1] + calib[p, 1:, 1:] - calib[p, :-1, 1:]
            ) / 2
            grady_calib[p, :-1, :-1] = (
                calib[p, :-1, 1:] - calib[p, :-1, :-1] + calib[p, 1:, 1:] - calib[p, 1:, :-1]
            ) / 2
        powder = np.sqrt(gradx_calib**2 + grady_calib**2)
    return powder

def generate_powder(powder_path, detname, smooth=False):
    """
    Generate the assembled powder plot and cache it.

    Parameters
    ----------
    powder_path : str
        Path to the h5 file containing the powder data.
    detname : str
        Name of the detector
    smooth : bool, optional
        If True, apply smoothing to the powder image.
    """
    powder = extract_powder(powder_path, detname)
    powder = preprocess_powder(powder, smooth)
    return powder

def build_detector(in_file, shape):
    """
    Read the metrology data and build a pyFAI detector object.

    Parameters
    ----------
    in_file : str
        Path to the input file
    shape : tuple
        Shape of the detector

    Returns
    -------
    pyFAI.Detector
        Configured pyFAI detector object
    """
    psana_to_pyfai = PsanaToPyFAI(
        in_file=in_file,
        shape=shape,
    )
    detector = psana_to_pyfai.detector
    return detector

def define_calibrant(calibrant, wavelength):
    """
    Define calibrant for optimization with appropriate wavelength

    Parameters
    ----------
    calibrant : str
        Name of the calibrant
    wavelength : float
        Wavelength of the experiment
    """
    calibrant = CALIBRANT_FACTORY(calibrant)
    calibrant.wavelength = wavelength
    return calibrant

def min_intensity(powder, threshold):
    """
    Estimates minimal intensity for extracting key Bragg peaks

    Parameters
    ----------
    powder : np.ndarray
        Powder image
    threshold : float
        Percentile for intensity thresholding
    """
    mean = np.mean(powder)
    std = np.std(powder)
    outlier = mean + 5 * std
    nice_pix = powder < outlier
    Imin = np.percentile(powder[nice_pix], threshold)
    powder = np.clip(powder, 0, outlier)
    return Imin