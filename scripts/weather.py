import numpy as np
from datetime import datetime, timedelta
import xarray as xr
from xarray.coding.times import SerializationWarning
from sklearn.preprocessing import StandardScaler
import sys, pathlib, warnings

warnings.filterwarnings("ignore", category=SerializationWarning)

RUN = "00"
DAY = (datetime.now() - timedelta(5)).strftime("%Y%m%d")
URL = f"http://nomads.ncep.noaa.gov/dods/gfs_0p25_1hr/gfs{DAY.replace('-','')}/gfs_0p25_1hr_{RUN}z"
VARIABLES = ["gustsfc", "dpt2m"]
DIR = "./data/weather/"

print("Weather Data Parameters")
print(f"Date: {DAY}")
print(f"Run Time: {RUN}z")
print(f"Variables: {VARIABLES}")
print(f"Data URL: {URL}")
print(f"Data Directory: {DIR}")
print("-" * 40)

path = pathlib.Path(DIR) / f"{DAY}_{RUN}z"
path.mkdir(parents=True, exist_ok=True)

try:
    print("Opening remote dataset...")
    ds = xr.open_dataset(URL, engine="netcdf4")
    print("Successfully accessed the dataset.")
except Exception as e:
    print("Failed to access the dataset.")
    print(f"Error: {e}")
    sys.exit(1)

# split
ds_train = ds.isel(time=list(range(0, 50, 2)))
ds_test  = ds.isel(time=list(range(1, 49, 2)))

# grid (shared)
lat = np.pi / 2 - np.deg2rad(ds_train.lat.values)  # colat
lon = np.deg2rad(ds_train.lon.values)

# time scaling (shared t0/min/max taken from train)
time_train_raw = ds_train.time.values
t0 = time_train_raw[0]
time_train = (time_train_raw - t0) / np.timedelta64(1, "h")
min_time, max_time = np.min(time_train), np.max(time_train)
time_train = (time_train - min_time) / (max_time - min_time)

time_test_raw = ds_test.time.values
time_test = (time_test_raw - t0) / np.timedelta64(1, "h")
time_test = (time_test - min_time) / (max_time - min_time)


train_pack = {
    "lat": lat,
    "lon": lon,
    "time": time_train,
    "t0": np.array(str(t0), dtype="U"), 
    "min_time": np.array(min_time),
    "max_time": np.array(max_time),
    "run": np.array(RUN, dtype="U"),
    "day": np.array(DAY, dtype="U"),
}
test_pack = {
    "lat": lat,
    "lon": lon,
    "time": time_test,
    "t0": np.array(str(t0), dtype="U"),
    "min_time": np.array(min_time),
    "max_time": np.array(max_time),
    "run": np.array(RUN, dtype="U"),
    "day": np.array(DAY, dtype="U"),
}

print("Extracting + standardizing variables...")
for var in VARIABLES:
    if var not in ds_train.variables:
        print(f"Variable not found: {var}")
        continue

    scaler = StandardScaler()

    # --- train ---
    data_train = ds_train[var].values
    sh_train = data_train.shape
    data_train_std = scaler.fit_transform(data_train.reshape(-1, 1)).reshape(sh_train)

    # --- test (use train scaler) ---
    data_test = ds_test[var].values
    sh_test = data_test.shape
    data_test_std = scaler.transform(data_test.reshape(-1, 1)).reshape(sh_test)

    # store standardized data + scaler params
    train_pack[f"{var}"] = data_train_std
    test_pack[f"{var}"] = data_test_std
    train_pack[f"{var}_mean"] = scaler.mean_.astype(np.float64)
    test_pack[f"{var}_mean"] = scaler.mean_.astype(np.float64)
    train_pack[f"{var}_scale"] = scaler.scale_.astype(np.float64)
    test_pack[f"{var}_scale"] = scaler.scale_.astype(np.float64)

    print(f"OK: {var} | mean={scaler.mean_[0]:.4g} scale={scaler.scale_[0]:.4g}")

# write one file per set
np.savez(path / "weather_train.npz", **train_pack)
np.savez(path / "weather_test.npz", **test_pack)

print("Saved:")
print(" -", path / "weather_train.npz")
print(" -", path / "weather_test.npz")

ds.close()
print("Done.")
