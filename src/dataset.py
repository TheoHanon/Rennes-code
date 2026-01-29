from typing import Literal, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

class WeatherDataset(Dataset):

    @classmethod
    def from_file(
        cls,
        path, 
        var : str = "dpt2m",
        dtype=torch.float32,
    ):
                
        data = np.load(path)
        t = torch.from_numpy(data["time"]).to(dtype) 
        lat = torch.from_numpy(data["lat"]).to(dtype)
        lon = torch.from_numpy(data["lon"]).to(dtype)
        y = torch.from_numpy(data[var]).to(dtype).reshape(-1, 1)

        t0 = data["t0"]
        min_time = data["min_time"]
        max_time = data["max_time"]
        mean = data[f"{var}_mean"]
        scale = data[f"{var}_scale"]

        return cls(t, lat, lon, y, t0, min_time, max_time, mean, scale)

    def __init__(
        self, 
        time : torch.Tensor, 
        lat : torch.Tensor,
        lon : torch.Tensor,
        y : torch.Tensor,
        t0 : float, 
        min_time : float, 
        max_time : float,
        mean : float,
        scale : float,
    ):
        
        super().__init__()

        self.t0 = t0
        self.min_time = min_time
        self.max_time = max_time
        self.mean = mean
        self.scale = scale

        self.time = time
        self.lat = lat
        self.lon = lon

        self.y = y
        times, lats, lons = torch.meshgrid(time, lat, lon, indexing = "ij")
        self.grid = torch.stack([times.ravel(), lats.ravel(), lons.ravel()], dim = -1)
        self.shape = (len(time), len(lat), len(lon))
    

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns grid and full field.
        """
        grid = torch.stack(self.grid, dim=-1).detach()  
        full_field = self.y.detach() * self.scale + self.mean
        return grid, full_field
    
    def get_shape(self) -> tuple:
        return self.shape

    def __len__(self):
        return self.y.size(0)
    
    def __getitem__(self, index):
        return self.grid[index], self.y[index]
    
    def __repr__(self) -> str:
        return (
            f"WeatherDataset\n"
            f"t0 = {self.t0}\n"
            f"min_time = {self.min_time}\n"
            f"max_time = {self.max_time}\n"
            f"Number of samples = {len(self)}\n"
        )
    



class GravityDataset(Dataset):

    @classmethod
    def from_file(
        cls,
        path,
        target: Literal["pot", "acc"],
        dtype=torch.float32,
        scale: float = 1e-2,
    ):
        data = np.load(path)
        # coords = torch.from_numpy(data["coord"]).reshape(-1, 3).to(dtype)
        rtp = torch.from_numpy(data["rtp"]).reshape(-1, 3).to(dtype)
        # pot already stores f = -Phi (dimensionless), multiply by GM to get physical
        pots = torch.from_numpy(data["pot"] * data["gm"]).reshape(-1, 1).to(dtype)
        accs = torch.from_numpy(data["acc"] * data["gm"]).reshape(-1, 3).to(dtype)

        rref = torch.tensor(data["rref"], dtype=dtype)
        gm = torch.tensor(data["gm"], dtype=dtype)
        L = int(data["L"])

        return cls(rtp, pots, accs, rref, gm, L, target, scale)

    def __init__(
        self,
        rtp: torch.Tensor,
        pots: torch.Tensor,
        accs: torch.Tensor,
        rref: torch.Tensor,
        gm: torch.Tensor,
        L: int,
        target: Literal["pot", "acc"],
        scale: float = 1e-2,
    ):
        super().__init__()

        self.rref = rref
        self.gm = gm
        self.target = target
        self.L = L
        self.scale = scale

        # Dimensionless coordinates
        self.coords = rtp
        self.coords[:, 0] /= rref
        self.r = self.coords[:, 0]

        # Monopole reference fields (dimensionless)
        monopole_pot = 1.0 / self.r.unsqueeze(-1)  # f0 = GM/|x| scaled by GM/rref
        a0_r = -1.0 / (self.r**2)  # radial component of ∇f0 scaled by GM/rref^2
        monopole_acc = np.column_stack((
            a0_r,
            np.zeros_like(a0_r),
            np.zeros_like(a0_r))
        )  # a0 = ∇f0 scaled by GM/rref^2

    
        if target == "acc":
            self.norm = gm / (rref * rref)  # GM / r_ref^2
            acc_dimless = accs / self.norm  # a / (GM/r_ref^2)
            # perturbation of acceleration
            self.targets = (acc_dimless - monopole_acc) / self.scale
            self._monopole = monopole_acc
        else:  # "pot"
            self.norm = gm / rref  # GM / r_ref
            pot_dimless = pots / self.norm  # f / (GM/r_ref)
            # perturbation of potential
            self.targets = (pot_dimless - monopole_pot) / self.scale
            self._monopole = monopole_pot

    def get(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns coords and normalized full field (not perturbation) on the (L+1, 2L+1) grid.
        """
        coords_grid = self.coords.reshape(self.L + 1, 2 * self.L + 1, 3).detach()
        full_field = (
            (self.targets * self.scale + self._monopole)
            .reshape(self.L + 1, 2 * self.L + 1, -1)
            .detach()
        )
        return coords_grid, full_field

    def __len__(self):
        return self.coords.size(0)

    def __getitem__(self, index):
        return self.coords[index], self.targets[index]

    def __repr__(self) -> str:
        rref = self.rref.item()
        gm = self.gm.item()
        norm = self.norm.item()
        unit = "m^2/s^2" if self.target == "pot" else "m/s^2"

        return (
            f"target = {self.target}\n"
            f"scaling = {self.scale}\n"
            f"L = {self.L}\n"
            f"rref = {rref:.5f} [m]\n"
            f"gm = {gm:.5f} [m^3/s^2]\n"
            f"norm = {norm:.5f} [{unit}]\n"
        )