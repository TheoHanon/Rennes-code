import numpy as np
from pathlib import Path
import pyshtools
from utils import *

# -----------------------
# Parameters
# -----------------------
LTRAIN = 120
LTEST = 120
DATADIR = "data/gravity/"
SHADR_FILE = "data/gravity/jgmro_120d_sha.tab"  # download from PDS

OUT_TRAIN = "gravity_train.npz"
OUT_TEST = "gravity_test.npz"

print("Gravity Data Parameters")
print(f"Train degree L: {LTRAIN}")
print(f"Test degree  L: {LTEST}")
print(f"Data dir: {DATADIR}")
print(f"SHADR file: {SHADR_FILE}")
print("-" * 40)

# -----------------------
# Paths + inputs
# -----------------------
path = Path(DATADIR)
path.mkdir(parents=True, exist_ok=True)

shadr_path = Path(SHADR_FILE)
if not shadr_path.exists():
    raise FileNotFoundError(f"SHADR file not found: {SHADR_FILE}")

# Read gravity model
cmn, rref, gm = read_shadr(SHADR_FILE)

# -----------------------
# Train set
# -----------------------
ntrain = LTRAIN + 1
npts_train = ntrain * (2 * ntrain - 1)

coords_train = np.random.randn(npts_train, 3)
norms = np.linalg.norm(coords_train, axis=-1, keepdims=True)
norms = np.maximum(norms, 1e-12)
coords_train *= (rref + 230e3) / norms  # altitude: +230 km

rtp_train = cart_to_rtp(coords_train)
pot_train, acc_sph_train = compute_pot_acc(rtp_train, cmn, rref)

train_pack = {
    "coord": coords_train,         # (N, 3)
    "rtp": rtp_train,              # (N, 3)
    "pot": pot_train,              # (N,)
    "acc": acc_sph_train,         # (N, 3)
    "rref": np.array(rref),
    "gm": np.array(gm),
    "L": np.array(LTRAIN, dtype=np.int32),
}

# -----------------------
# Test set
# -----------------------
ntest = LTEST + 1

lat, phi = pyshtools.expand.GLQGridCoord(LTEST)
lat = np.deg2rad(lat)
phi = np.deg2rad(phi)
theta = np.pi / 2 - lat  # colat

thetas, phis = np.meshgrid(theta, phi, indexing="ij")

rs = 1.1 * rref * np.ones_like(thetas)  # radius shell
rtp_test = np.stack([rs.ravel(), thetas.ravel(), phis.ravel()], axis=-1)
coords_test = rtp_to_cart(rtp_test)

pot_test, acc_sph_test = compute_pot_acc(rtp_test, cmn, rref)

# reshape to grid layout (same as your original)
coords_test = coords_test.reshape(ntest, 2 * ntest - 1, 3)
rtp_test = rtp_test.reshape(ntest, 2 * ntest - 1, 3)
pot_test = pot_test.reshape(ntest, 2 * ntest - 1)
acc_sph_test = acc_sph_test.reshape(ntest, 2 * ntest - 1, 3)

test_pack = {
    "coord": coords_test,          # (ntest, 2*ntest-1, 3)
    "rtp": rtp_test,               # (ntest, 2*ntest-1, 3)
    "pot": pot_test,               # (ntest, 2*ntest-1)
    "acc": acc_sph_test,          # (ntest, 2*ntest-1, 3)
    "rref": np.array(rref),
    "gm": np.array(gm),
    "L": np.array(LTEST, dtype=np.int32),
}

# -----------------------
# Save: one file per set
# -----------------------
np.savez(path / OUT_TRAIN, **train_pack)
np.savez(path / OUT_TEST, **test_pack)

print("Saved:")
print(" -", path / OUT_TRAIN)
print(" -", path / OUT_TEST)
print("Done.")
