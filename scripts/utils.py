import numpy as np
import pyshtools

from rich.progress import track


def read_shadr(path: str):
    """
    Read SHADR file and return SH coefficients plus reference radius and GM.
    """
    meta = np.loadtxt(
        path, delimiter=",", max_rows=1, usecols=(0, 1, 3), dtype=np.float64
    )
    data = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(0, 1, 2, 3))

    rref = 1e3 * meta[0]  # m
    gm = 1e9 * meta[1]  # m^3 / s^2
    Lmax = int(meta[2])

    # cmn[0] = C_nm, cmn[1] = S_nm
    cmn = np.zeros((2, Lmax + 1, Lmax + 1), dtype=np.float64)
    cmn[0, 0, 0] = 1.0

    n = data[:, 0].astype(np.int32)
    m = data[:, 1].astype(np.int32)

    cmn[0, n, m] = data[:, 2]
    cmn[1, n, m] = data[:, 3]

    return cmn, rref, gm


def cart_to_rtp(coords: np.ndarray) -> np.ndarray:
    """
    Cartesian (x, y, z) -> (r, theta, phi) in NumPy.
    theta: colatitude, phi: longitude.
    """
    x, y, z = coords[..., 0], coords[..., 1], coords[..., 2]
    r = np.linalg.norm(coords, axis=-1)
    r = np.maximum(r, 1e-8)  # avoid division by zero

    theta = np.arccos(np.clip(z / r, -1.0, 1.0)) # [0, pi]
    phi = np.mod(np.arctan2(y, x), 2*np.pi) # [0, 2pi)

    return np.stack((r, theta, phi), axis=-1)


def rtp_to_cart(rtp: np.ndarray) -> np.ndarray:

    r, theta, phi = rtp[..., 0], rtp[..., 1], rtp[..., 2]
    sin_theta = np.sin(theta)

    x = r * sin_theta * np.cos(phi)
    y = r * sin_theta * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=-1)


def sph_field_to_cart(rtp: np.ndarray, field: np.ndarray) -> np.ndarray:
    """
    Convert vector field from spherical basis (r, theta, phi)
    to Cartesian basis (x, y, z) in NumPy.
    """
    theta = rtp[..., 1]
    phi = rtp[..., 2]

    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    fr = field[..., 0]
    fth = field[..., 1]
    fphi = field[..., 2]

    xf = fr * sin_theta * cos_phi + fth * cos_theta * cos_phi - fphi * sin_phi
    yf = fr * sin_theta * sin_phi + fth * cos_theta * sin_phi + fphi * cos_phi
    zf = fr * cos_theta - fth * sin_theta

    return np.stack((xf, yf, zf), axis=-1)


def compute_pot_acc(
    rtp: np.ndarray,
    cmn: np.ndarray,
    rref: float,
):
    """Computes the gravitational potentials and acceleration for the Stokes coefficients in the matrix coeffs and at the points of spherical coordinates in v

    Args:
        coeffs (np.array): matrix of coefficients of size (2,n_max+1, n_max+1)
        r_0: reference radius (m)
        v (np.array): spherical coordinates (r, theta, phi) of the msr points

    Returns:
        pot (np.array): potential divided by gm
        acc (np.array): acceleration divided by gm
    """

    n_max = cmn.shape[1] - 1  # deg max of the SH

    # reading spherical coordinates
    r = rtp[:, 0]
    theta = rtp[:, 1]
    lbd = rtp[:, 2]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # matrix of 4-pi-normalized associated Legendre polynomials and their first derivatives at each point
    Pmn_full = np.array([pyshtools.legendre.PlmBar_d1(n_max + 1, z) for z in cos_theta])
    # separating polynomials (Pmn_z) and their derivatives (dPmn_z)
    Pmn_z = Pmn_full[:, 0, ...]
    dPmn_z = Pmn_full[:, 1, ...]

    u = np.zeros(r.size)  # vector of potential msr
    g = np.zeros((r.size, 3))  # matrix of acceleration msr

    for n in track(range(n_max + 1)):
        t1 = (rref / r) ** n
        for m in range(n + 1):
            # index in last dimension of Pmn matrix (see PlmBar_d1 docs)
            pmn_idx = (n * (n + 1)) // 2 + m

            # contribution of Cnm to potential
            u += (
                t1
                * Pmn_z[:, pmn_idx]
                * (cmn[0, n, m] * np.cos(m * lbd) + cmn[1, n, m] * np.sin(m * lbd))
            )

            # contribution of Cnm to accleration components (in spherical coordinates)
            g[:, 0] += (
                -t1
                * (n + 1)
                * Pmn_z[:, pmn_idx]
                * (cmn[0, n, m] * np.cos(m * lbd) + cmn[1, n, m] * np.sin(m * lbd))
            )
            g[:, 1] += (
                t1
                * dPmn_z[:, pmn_idx]
                * (-sin_theta)
                * (cmn[0, n, m] * np.cos(m * lbd) + cmn[1, n, m] * np.sin(m * lbd))
            )
            g[:, 2] += (
                t1
                / sin_theta
                * Pmn_z[:, pmn_idx]
                * m
                * (-cmn[0, n, m] * np.sin(m * lbd) + cmn[1, n, m] * np.cos(m * lbd))
            )

    return u / r, g / (r**2)[:, None]