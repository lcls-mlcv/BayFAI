import numpy as np

def rotation_matrix(params):
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
    # Rotation about axis 1: Note this rotation is left-handed
    rot1 = np.array([[1.0, 0.0, 0.0],
                        [0.0, cos_rot1, sin_rot1],
                        [0.0, -sin_rot1, cos_rot1]])
    # Rotation about axis 2. Note this rotation is left-handed
    rot2 = np.array([[cos_rot2, 0.0, -sin_rot2],
                        [0.0, 1.0, 0.0],
                        [sin_rot2, 0.0, cos_rot2]])
    # Rotation about axis 3: Note this rotation is right-handed
    rot3 = np.array([[cos_rot3, -sin_rot3, 0.0],
                        [sin_rot3, cos_rot3, 0.0],
                        [0.0, 0.0, 1.0]])
    rotation_matrix = np.dot(np.dot(rot3, rot2), rot1)  # 3x3 matrix
    return rotation_matrix

def correct_geom(detector, params):
    """
    Correct the geometry based on the given parameters found by PyFAI calibration
    """
    p1, p2, p3 = detector.calc_cartesian_positions()
    dist = params[0]
    poni1 = params[1]
    poni2 = params[2]
    p1 = (p1 - (detector.pixel_size / 2) - poni1).ravel()
    p2 = (p2 - (detector.pixel_size / 2) - poni2).ravel()
    if p3 is None:
        p3 = np.zeros_like(p1) + dist
    else:
        p3 = (p3+dist).ravel()
    coord_det = np.vstack((p1, p2, p3))
    coord_sample = np.dot(rotation_matrix(params), coord_det)
    x, y, z = coord_sample
    x = np.reshape(x, detector.raw_shape)
    y = np.reshape(y, detector.raw_shape)
    z = np.reshape(z, detector.raw_shape)
    return x, y, z

def calculate_2theta(detector, params):
    """
    Calculate the 2theta angles for the detector based on the geometry parameters.

    Parameters
    ----------
    detector : pyFAI.detectors.Detector
        PyFAI detector object containing pixel index map and shape information.
    params : list
        6 Geometry parameters: distance, x-shift, y-shift, Rx, Ry, Rz
    """
    x, y, z = correct_geom(detector, params)
    ttha = np.zeros(detector.raw_shape)
    # loop through the panels
    for p in range(detector.n_modules):
        ttha[p, :] = np.arctan2(np.sqrt(x[p]*x[p]+y[p]*y[p]), z[p])
    return ttha

def get_radius_map(detector, center=None):
    """
    Compute each pixel's radius for an array with input shape and center.

    Parameters
    ----------
    detector  : pyFAI Detector object
        detector object containing pixel infos
    center : 2d tuple or array
        (cx,cy) detector center in meters; if None, choose image center

    Returns
    -------
    r : numpy.ndarray, with input shape
        map of pixels' radii
    """
    y, x, z = detector.calc_cartesian_positions()
    if center is None:
        center = (0, 0)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    return r

def radial_profile(powder, detector, center=None):
    """
    Compute the radial intensity profile of an image.

    Parameters
    ----------
    powder : numpy.ndarray, shape (n,m)
        detector image
    center : 2d tuple or array
        (cx,cy) beam center in meter; if None, choose detector origin

    Returns
    -------
    radialprofile : numpy.ndarray, 1d
        radial intensity profile of input image
    """
    if center is None:
        center = (0, 0)
    r = get_radius_map(detector, center=center)
    intensity, bin_edges = np.histogram(
        r.ravel(), bins=1000, range=(r.min(), r.max()), weights=powder.ravel()
    )
    count, _ = np.histogram(r.ravel(), bins=bin_edges)
    radialprofile = np.divide(
        intensity, count, out=np.zeros_like(intensity), where=count != 0
    )
    r_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return radialprofile, r_centers

def pix2q(pixels, distance, wavelength):
    """
    Convert distance from number of pixels from detector center to q-space.

    Parameters
    ----------
    pixels : numpy.ndarray, 1d
        distance in meter from beam center
    distance : float
        detector distance in meter
    Returns
    -------
    qvals : numpy.ndarray, 1d
        magnitude of q-vector in per Angstrom
    """
    theta = np.arctan2(pixels, distance)
    return 4.0 * np.pi * np.sin(theta / 2.0) / (wavelength * 1e10)