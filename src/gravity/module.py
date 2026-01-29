import spherical_inr as sph

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from typing import List, Tuple

class GravityPE(sph.HerglotzPE):

    def forward(self, x : torch.Tensor) -> torch.Tensor :
        """Input: x = (r, \theta, \phi) """
        r = x[..., 0].unsqueeze(-1)
        xyz = sph.rtp_to_r3(x)
        f = 1/r * super().forward(xyz / r.pow(2))
        return f

class GravityModule(nn.Module):

    def __init__(
        self,
        num_atoms: int,
        mlp_sizes: List[int],
        output_dim: int,
        *,
        bias: bool = False,
        L_init: int = 15,
        omega0_mlp: float = 1.0,
        rot: bool = False,
    ):

        super().__init__()
        self.pe = GravityPE(num_atoms=num_atoms, L_init=L_init, rot=rot)
        self.mlp = sph.SineMLP(
            input_features=num_atoms,
            output_features=output_dim,
            hidden_sizes=mlp_sizes,
            bias=bias,
            omega0=omega0_mlp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Input: x = (r, \theta, \phi)"""
        pe_out = self.pe(x)
        out = self.mlp(pe_out)
        return out
    

class LitGravityModule(pl.LightningModule):

    def __init__(
        self, 
        num_atoms: int,
        mlp_sizes: List[int],
        output_dim: int,
        *,
        bias: bool = False,
        L_init: int = 15,
        omega0_mlp: float = 1.0,
        rot: bool = False,
        lr : float = 1e-3,
        reg : bool = True,
        lam : float = 1e-3,
        Nreg : int = 5000,
        Rrange : Tuple[float, float] = (1.0, 4.0)
    ): 
        
        super().__init__()
        self.save_hyperparameters()

        self.model = GravityModule(
            num_atoms=num_atoms,
            mlp_sizes = mlp_sizes,
            output_dim=output_dim,
            bias = bias,
            L_init=L_init,
            omega0_mlp = omega0_mlp,
            rot = rot,
        )

        self.lr = lr
        self.reg = reg
        self.lam = lam
        self.Nreg = Nreg
        self.Rrange = Rrange


    def forward(self, x: torch.Tensor):
        return self.model(x)
    
    def _r3_uniform(
        self,
    ) -> torch.Tensor:

        u = torch.rand(self.Nreg)
        rmin_3 = self.Rrange[0] ** 3
        rmax_3 = self.Rrange[1] ** 3
        radii = ((rmax_3 - rmin_3) * u + rmin_3).pow(1 / 3)

        cos_theta = torch.empty(self.Nreg).uniform_(-1, 1)
        theta = torch.acos(cos_theta)
        phi = torch.empty(self.Nreg).uniform_(0, 2 * torch.pi)

        return torch.stack([radii, theta, phi], axis = -1)


    def on_train_epoch_start(self) -> None:
        super().on_train_epoch_start()
        if self.reg:
            self.xreg_epoch = self._r3_uniform().to(self.device)


    def training_step(self, batch, batch_idx):
        x, acc = batch
        # ----- MAIN LOSS -----
        x = x.clone().detach().requires_grad_(True)
        yhat = self.forward(x)
        grad = sph.spherical_gradient(yhat, x, track = True)
        loss_data = F.mse_loss(grad, acc)
        loss = loss_data
        # ----- REGULARIZER -----
        if self.reg:
            xreg = self.xreg_epoch.detach().clone().requires_grad_(True)
            yreg = self.forward(xreg)
            grad_reg = sph.spherical_gradient(yreg, xreg, track = True)
            lap_reg = sph.spherical_divergence(grad_reg, xreg, track = True).pow(2).mean()
            loss += self.lam * lap_reg
            self.log("loss_reg", self.lam * lap_reg, prog_bar=True, on_step=True, on_epoch=True)

        self.log(
            "loss_data",
            loss_data,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optim




