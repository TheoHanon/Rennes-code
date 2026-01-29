import spherical_inr as sph

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import List, Tuple, Optional

def fourier_kernel(
    x: torch.Tensor,
    weight: torch.Tensor,
    omega0: float,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    
    z = omega0 * F.linear(x,  weight, bias)
    return torch.exp(1j * z)


def herglotz_kernel(
    x: torch.Tensor,
    A_real: torch.Tensor,
    A_imag: torch.Tensor,
    sigma: torch.Tensor,
    inv_const: torch.Tensor,
):

    ax_R = F.linear(x, A_real)  # (..., num_atoms)
    ax_I = F.linear(x, A_imag)

    ax = ax_R + 1j * ax_I
    rho = F.softplus(sigma)
    z = (1.0 + 2.0 * rho * ax) * torch.exp(rho * (ax - 1.0))

    return inv_const * z


class ComplexHerglotzPE(sph.HerglotzPE):

    def __init__(self, num_atoms : int, L_init : int):
        super().__init__(num_atoms, L_init, rot = False)

    def forward(self, x:torch.Tensor):
        return herglotz_kernel(
            x, 
            self.A_real0, 
            self.A_imag0,
            self.sigmas, 
            self.inv_const
        )


class WeatherPE(ComplexHerglotzPE):

    def __init__(self, num_atoms : int, L_init : int, omega0: float = 30.0):
        super().__init__(num_atoms, L_init)

        self.omega0 = omega0
        self.Omega = nn.Parameter(torch.empty((self.num_atoms, 1)).uniform_(-1, 1))
        self.phase = nn.Parameter(torch.zeros((self.num_atoms)))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input is (t, theta, phi)"""
        t = x[..., 0].unsqueeze(-1)
        xyz = sph.tp_to_r3(x[..., 1:])
        
        fourier = fourier_kernel(t, self.Omega, self.omega0, self.phase)
        herglotz = super().forward(xyz)

        out = fourier.real * herglotz.real - fourier.imag * herglotz.imag
        return out

class WeatherModule(nn.Module):

    def __init__(
        self,
        num_atoms: int,
        mlp_sizes: List[int],
        output_dim: int,
        *,
        bias: bool = True,
        L_init: int = 15,
        omega0_mlp: float = 1.0,
        omega0_pe : float = 30.0,
    ):

        super().__init__()
        self.pe = WeatherPE(num_atoms=num_atoms, L_init=L_init, omega0=omega0_pe)
        self.mlp = sph.SineMLP(
            input_features=num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            bias=bias,
            omega0=omega0_mlp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: x = (t, \theta, \phi)"""
        pe_out = self.pe(x)
        out = self.mlp(pe_out)
        return out
    

class LitWeatherModule(pl.LightningModule):

    def __init__(
        self, 
        num_atoms: int,
        mlp_sizes: List[int],
        output_dim: int,
        *,
        bias: bool = True,
        L_init: int = 15,
        omega0_mlp: float = 1.0,
        omega0_pe: float = 30.0,
        lr : float = 1e-3,
    ): 
        
        super().__init__()
        self.save_hyperparameters()

        self.model = WeatherModule(
            num_atoms=num_atoms,
            mlp_sizes = mlp_sizes,
            output_dim=output_dim,
            bias = bias,
            L_init=L_init,
            omega0_mlp = omega0_mlp,
            omega0_pe = omega0_pe,
        )

        self.lr = lr
    

    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        # ----- MAIN LOSS -----
        x = x.clone().detach().requires_grad_(True)
        yhat = self.forward(x)
        loss = F.mse_loss(yhat, y)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim










    