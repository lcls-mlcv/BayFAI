import numpy as np
import numpy.typing as npt
from typing import Optional
import pyFAI
from scipy.ndimage import gaussian_filter1d

def rotation_matrix(params: list) -> np.ndarray:
    """
    Compute and return the detector tilts as a single rotation matrix

    Parameters
    ----------
    params : list
        Detector parameters found by PyFAI calibration
    """
    cos_rot1 = np.cos(params[3])
    cos_rot2 = np.cos(params[4])
    cos_rot3 = np.cos(params[5])
    sin_rot1 = np.sin(params[3])
    sin_rot2 = np.sin(params[4])
    sin_rot3 = np.sin(params[5])
    # Rotation about vertical axis: Note this rotation is left-handed
    rot1 = np.array(
        [[1.0, 0.0, 0.0], [0.0, cos_rot1, sin_rot1], [0.0, -sin_rot1, cos_rot1]]
    )
    # Rotation about horizontal axis: Note this rotation is left-handed
    rot2 = np.array(
        [[cos_rot2, 0.0, -sin_rot2], [0.0, 1.0, 0.0], [sin_rot2, 0.0, cos_rot2]]
    )
    # Rotation about z-axis: Note this rotation is right-handed
    rot3 = np.array(
        [[cos_rot3, -sin_rot3, 0.0], [sin_rot3, cos_rot3, 0.0], [0.0, 0.0, 1.0]]
    )
    rotation_matrix = np.dot(np.dot(rot3, rot2), rot1)
    return rotation_matrix


def correct_geom(detector: pyFAI.detectors.Detector, params: Optional[list] = None):
    """
    Correct the geometry given a set of geometry parameters.

    Parameters
    ----------
    detector : pyFAI.detectors.Detector
        PyFAI detector object containing pixel coordinates.
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
    """
    x, y, z = detector.calc_cartesian_positions()
    if params is not None:
        dist = params[0]
        poni1 = params[1]
        poni2 = params[2]
        x = (x - poni1).ravel()
        y = (y - poni2).ravel()
        if z is None:
            z = np.zeros_like(x) + dist
        else:
            z = (z + dist).ravel()
        coord_det = np.vstack((x, y, z))
        x, y, z = np.dot(rotation_matrix(params), coord_det)
    x = np.reshape(x, detector.raw_shape)
    y = np.reshape(y, detector.raw_shape)
    z = np.reshape(z, detector.raw_shape)
    return x, y, z


def calculate_radius(
    detector: pyFAI.detectors.Detector, params: Optional[list] = None
) -> np.ndarray:
    """
    Calculate the radius for each pixel based on the geometry parameters.

    Parameters
    ----------
    detector  : pyFAI.Detector
        pyFAI detector object
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz

    Returns
    -------
    r : numpy.ndarray, with input shape
        map of pixels' radii
    """
    x, y, _ = correct_geom(detector, params)
    r = np.zeros(detector.raw_shape)
    for p in range(detector.n_modules):
        r[p] = np.sqrt(x[p] ** 2 + y[p] ** 2)
    return r


def calculate_2theta(
    detector: pyFAI.detectors.Detector, params: Optional[list] = None
) -> np.ndarray:
    """
    Calculate the 2θ angles for the detector based on the geometry parameters.

    Parameters
    ----------3
    detector : pyFAI.detectors.Detector
        PyFAI detector object containing pixel coordinates to be corrected.
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
    """
    x, y, z = correct_geom(detector, params)
    tth = np.zeros(detector.raw_shape)
    for p in range(detector.n_modules):
        tth[p] = np.arctan2(np.sqrt(x[p] * x[p] + y[p] * y[p]), z[p])
    return tth


def calculate_q(
    detector: pyFAI.detectors.Detector, params: Optional[list] = None
) -> np.ndarray:
    """
    Calculate the q-vectors for each pixel based on the geometry parameters.

    Parameters
    ----------
    detector : pyFAI.detectors.Detector
        PyFAI detector object containing pixel coordinates to be corrected.
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
    """
    tth = calculate_2theta(detector, params)
    wavelength = detector.wavelength
    q = 4.0 * np.pi * np.sin(tth / 2.0) / (wavelength * 1e10)
    return q


def azimuthal_integration(
    powder: npt.NDArray[np.float64],
    detector: pyFAI.detectors.Detector,
    params: Optional[list] = None,
) -> tuple:
    """
    Compute the radial intensity profile of an image.

    Parameters
    ----------
    powder : numpy.ndarray, shape (n,m)
        detector image
    detector : pyFAI.Detector
        PyFAI detector object
    params : list, optional
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
    """
    tth = calculate_2theta(detector, params)
    n_bins = round(len(tth.ravel()) / 4000) # aim for ~4000 pixels per bin
    intensity, bin_edges = np.histogram(
        tth.ravel(), bins=n_bins, range=(tth.min(), tth.max()), weights=powder.ravel()
    )
    count, _ = np.histogram(tth.ravel(), bins=bin_edges)
    radialprofile = np.divide(
        intensity, count, out=np.zeros_like(intensity), where=count != 0
    )
    radialprofile = gaussian_filter1d(radialprofile, sigma=2)
    tth_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return radialprofile, tth_centers


def theta2q(theta: np.ndarray, wavelength: float) -> np.ndarray:
    """
    Convert pixel 2θ angles to scattering vector magnitude q.

    Parameters
    ----------
    theta: : numpy.ndarray, 1d
        diffraction angles in radians
    wavelength : float
        X-ray wavelength in Angstrom

    Returns
    -------
    qs: numpy.ndarray, 1d
        magnitude of q-vector in per Angstrom
    """
    qs = 4.0 * np.pi * np.sin(theta / 2.0) / (wavelength * 1e10)
    return qs


def r2q(radii: np.ndarray, distance: float, wavelength: float) -> np.ndarray:
    """
    Convert pixel radii to scattering vector magnitude q.

    Parameters
    ----------
    radii : numpy.ndarray, 1d
        radius in meter from beam center
    distance : float
        detector distance in meter
    wavelength : float
        X-ray wavelength in Angstrom

    Returns
    -------
    qs: numpy.ndarray, 1d
        magnitude of q-vector in per Angstrom
    """
    theta = np.arctan2(radii, distance)
    qs = theta2q(theta, wavelength)
    return qs