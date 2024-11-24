from contextlib import contextmanager, nullcontext

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import diffusers

from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    look_at_view_transform,
)
from pytorch3d.renderer.camera_utils import join_cameras_as_batch

from monai.utils import optional_import
from monai.networks.nets import DiffusionModelUNet
from monai.losses.perceptual import PerceptualLoss
from monai.metrics import PSNRMetric, SSIMMetric

tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

from typing import Any, Callable, Dict, Optional, OrderedDict, Tuple, List
from lightning.pytorch import LightningModule
from omegaconf import DictConfig, OmegaConf

from dvr.renderer import ObjectCentricXRayVolumeRenderer

def make_cameras_dea(
    dist: torch.Tensor,
    elev: torch.Tensor,
    azim: torch.Tensor,
    fov: int = 40,
    znear: int = 4.0,
    zfar: int = 8.0,
    is_orthogonal: bool = False,
):
    assert dist.device == elev.device == azim.device
    _device = dist.device
    R, T = look_at_view_transform(dist=dist, elev=elev * 90, azim=azim * 180)
    if is_orthogonal:
        return FoVOrthographicCameras(R=R, T=T, znear=znear, zfar=zfar).to(_device)
    return FoVPerspectiveCameras(R=R, T=T, fov=fov, znear=znear, zfar=zfar).to(_device)

    
class FoVLightningModule(LightningModule):
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig):
        super().__init__()

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        
        self.fwd_renderer = ObjectCentricXRayVolumeRenderer(
            image_width=model_cfg.img_shape,
            image_height=model_cfg.img_shape,
            n_pts_per_ray=model_cfg.n_pts_per_ray,
            min_depth=model_cfg.min_depth,
            max_depth=model_cfg.max_depth,
            ndc_extent=model_cfg.ndc_extent,
        )

        # self.unet2d_model = DiffusionModelUNet(
        #     spatial_dims=2,
        #     in_channels=1,
        #     out_channels=model_cfg.fov_depth,
        #     num_res_blocks = (2, 2, 2, 2),
        #     channels = (256, 256, 512, 512),
        #     attention_levels = (False, False, True, True),
        #     norm_num_groups = 32,
        #     norm_eps = 1e-6,
        #     resblock_updown = True,
        #     num_head_channels = 8,
        #     with_conditioning = True,
        #     transformer_num_layers = 1,
        #     cross_attention_dim = 12,
        #     num_class_embeds = None,
        #     upcast_attention = True,
        #     dropout_cattn = 0.5,
        #     include_fc = True,
        #     use_combined_linear = True,
        #     use_flash_attention = True,
        # )

        # self.unet3d_model = None
        self.unet2d_model = diffusers.models.UNet2DConditionModel(
            sample_size=model_cfg.img_shape, 
            in_channels=1, 
            out_channels=model_cfg.fov_depth, 
            layers_per_block=2, # how many ResNet layers to use per UNet block
            block_out_channels=(256, 256, 512, 512), # the number of output channels for eaxh UNet block
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "CrossAttnDownBlock2D",
                "CrossAttnDownBlock2D",
            ),
            up_block_types=(
                "CrossAttnUpBlock2D", 
                "CrossAttnUpBlock2D", 
                "UpBlock2D",
                "UpBlock2D"
            ),
            dropout=0.5,
            cross_attention_dim=12
        )

        self.unet3d_model = None 
        # self.unet3d_model = diffusers.models.AutoencoderKLCogVideoX(
        #     sample_height=model_cfg.img_shape, 
        #     sample_width=model_cfg.img_shape, 
        #     in_channels=1, 
        #     out_channels=1, 
        #     layers_per_block=2, # how many ResNet layers to use per UNet block
        #     block_out_channels=(128, 256), # the number of output channels for eaxh UNet block
        #     down_block_types=(
        #         "CogVideoXDownBlock3D",
        #         "CogVideoXDownBlock3D",
        #         # "CogVideoXDownBlock3D",
        #         # "CogVideoXDownBlock3D"
        #     ),
        #     up_block_types=(
        #         # "CogVideoXUpBlock3D",
        #         # "CogVideoXUpBlock3D",
        #         "CogVideoXUpBlock3D",
        #         "CogVideoXUpBlock3D"
        #     ),
        # ) 
        
        # self.perc25d_loss = None
        self.perc25d_loss = PerceptualLoss(
            spatial_dims=3, 
            network_type="radimagenet_resnet50", 
            is_fake_3d=True, 
            fake_3d_ratio=16/256.
        ).eval()

        self.perc30d_loss = None
        # self.perc30d_loss = PerceptualLoss(
        #     spatial_dims=3, 
        #     network_type="medicalnet_resnet50_23datasets", 
        #     is_fake_3d=False, 
        #     # fake_3d_ratio=10/256.
        # ).eval()

        if model_cfg.phase=="finetune":
            pass
        
        if self.train_cfg.ckpt:
            print("Loading.. ", self.train_cfg.ckpt)
            checkpoint = torch.load(self.train_cfg.ckpt, map_location=torch.device("cpu"))["state_dict"]
            state_dict = {k: v for k, v in checkpoint.items() if k in self.state_dict()}
            self.load_state_dict(state_dict, strict=False)
            # self.load_from_checkpoint(self.train_cfg.ckpt)

        self.save_hyperparameters()
        self.train_step_outputs = []
        self.validation_step_outputs = []

        self.psnr = PSNRMetric(max_val=1.0)
        self.ssim = SSIMMetric(spatial_dims=3, data_range=1.0)
        self.psnr_outputs = []
        self.ssim_outputs = []

    def correct_window(self,
        T_old, 
        a_min=-1024, 
        a_max=3071, 
        b_min=-512, #-100, #-512, 
        b_max=3071, #+900, #3071, #1536, #2560, 
        factor=1.0    
    ):
        # Calculate the range for the old and new scales
        range_old = a_max - a_min
        range_new = b_max - b_min

        # Reverse the incorrect scaling
        T_raw = (T_old * range_old) + a_min

        # Apply the correct scaling
        T_new = (T_raw - b_min) / range_new 
        return T_new.clamp(0, 1)
    
    def forward_screen(self, image3d, cameras, learnable_windowing=False):
        image3d = self.correct_window(image3d, a_min=-1024, a_max=3071, b_min=-512, b_max=3071)
        image2d = self.fwd_renderer(image3d.float(), cameras.float()).clamp_(0, 1)
        return image2d
    
    def flatten_cameras(self, cameras, zero_translation=False):
        camera_ = cameras.clone()
        R = camera_.R
        if zero_translation:
            T = torch.zeros_like(camera_.T.unsqueeze_(-1))
        else:
            T = camera_.T.unsqueeze_(-1)
        return torch.cat([R.reshape(-1, 1, 9), T.reshape(-1, 1, 3)], dim=-1).contiguous().view(-1, 1, 12)

    def forward_volume(self, image2d, cameras, is_training=False):
        image2d = torch.flip(image2d, dims=(-1,))
        # out = self.unet2d_model.forward(image2d)
        _device = image2d.device
        B = image2d.shape[0]
        timesteps = 0*torch.randint(0, 1000, (B,), device=_device).long()  
        cam_feats = self.flatten_cameras(cameras)
        out = self.unet2d_model.forward(image2d, timesteps, cam_feats).sample

        # Resample the frustum out
        z = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=_device)
        y = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=_device)
        x = torch.linspace(-1.0, 1.0, steps=self.model_cfg.vol_shape, device=_device)
        grd = torch.stack(torch.meshgrid(x, y, z), dim=-1).view(-1, 3).unsqueeze(0).repeat(image2d.shape[0], 1, 1)  # 1 DHW 3 to B DHW 3
        
        # Process (resample) the volumes from ray views to ndc
        pts = cameras.transform_points_ndc(grd)  # world to ndc, 1 DHW 3
        res = F.grid_sample(
            out.float().unsqueeze(1), 
            pts.view(-1, self.model_cfg.vol_shape, self.model_cfg.vol_shape, self.model_cfg.vol_shape, 3).float(), 
            mode="bilinear", 
            padding_mode="zeros", 
            align_corners=True,
        )
        # res = torch.permute(res, [0, 1, 4, 2, 3])
        res = torch.permute(res, [0, 1, 4, 3, 2])
        res = torch.flip(res, dims=(-2,))

        if self.unet3d_model is not None:
            # dist_hidden = 8 * torch.ones(B, device=_device)
            # elev_hidden = torch.zeros(B, device=_device)
            # azim_hidden = torch.zeros(B, device=_device)
            # view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=self.model_cfg.fov, znear=self.model_cfg.min_depth, zfar=self.model_cfg.max_depth)
            # out = self.unet3d_model(res, timesteps, self.flatten_cameras(view_hidden))
            out = self.unet3d_model(res)
            if out.shape[1]==3:
                out = torch.cat([
                    torch.permute(out[:,[0]], [0, 1, 2, 3, 4]), 
                    torch.permute(out[:,[1]], [0, 1, 3, 4, 2]), 
                    torch.permute(out[:,[2]], [0, 1, 4, 2, 3]),
                ], dim=1).mean(dim=1, keepdim=True)

            # out = self.unet3d_model(res, timesteps=timesteps)
            if is_training:
                prob = torch.rand(1).item()
                if prob < 0.1:
                    return res
                else:
                    return out
            else:
                return out
            return out
            # return torch.cat([out, res], dim=1)
        else:
            return res
    
    def _common_step(self, batch, batch_idx, stage: Optional[str] = "evaluation"):
        image2d = batch["image2d"]
        image3d = batch["image3d"]
        _device = batch["image3d"].device
        B = image2d.shape[0]

        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_random = 8 * torch.ones(B, device=_device)
        # elev_random = torch.zeros(B, device=_device)
        elev_random = torch.rand_like(dist_random) - 0.5
        azim_random = torch.rand_like(dist_random) * 2 - 1  # from [0 1) to [-1 1)
        view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=self.model_cfg.fov, znear=self.model_cfg.min_depth, zfar=self.model_cfg.max_depth)
    
        dist_hidden = 8 * torch.ones(B, device=_device)
        elev_hidden = torch.zeros(B, device=_device)
        azim_hidden = torch.zeros(B, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=self.model_cfg.fov, znear=self.model_cfg.min_depth, zfar=self.model_cfg.max_depth)

        # Construct the samples in 2D
        figure_xr_source_hidden = image2d
        figure_ct_source_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)
        figure_ct_source_random = self.forward_screen(image3d=image3d, cameras=view_random)
            
        # Run the forward pass
        figure_dx_source_concat = torch.cat([figure_xr_source_hidden, figure_ct_source_hidden, figure_ct_source_random])
        camera_dx_render_concat = join_cameras_as_batch([view_hidden, view_hidden, view_random])

        # For 3D
        volume_dx_reproj_concat = self.forward_volume(
            image2d=figure_dx_source_concat, 
            cameras=camera_dx_render_concat, 
        )
        volume_xr_reproj_hidden, \
        volume_ct_reproj_hidden, \
        volume_ct_reproj_random = torch.split(volume_dx_reproj_concat, B, dim=0)
           
        figure_xr_reproj_hidden_hidden = self.forward_screen(image3d=volume_xr_reproj_hidden[:,[0],...], cameras=view_hidden)
        figure_xr_reproj_hidden_random = self.forward_screen(image3d=volume_xr_reproj_hidden[:,[0],...], cameras=view_random)
        
        figure_ct_reproj_hidden_hidden = self.forward_screen(image3d=volume_ct_reproj_hidden[:,[0],...], cameras=view_hidden)
        figure_ct_reproj_hidden_random = self.forward_screen(image3d=volume_ct_reproj_hidden[:,[0],...], cameras=view_random)
        
        figure_ct_reproj_random_hidden = self.forward_screen(image3d=volume_ct_reproj_random[:,[0],...], cameras=view_hidden)
        figure_ct_reproj_random_random = self.forward_screen(image3d=volume_ct_reproj_random[:,[0],...], cameras=view_random)
        
        im3d_loss_inv = F.l1_loss(volume_ct_reproj_hidden, image3d) * 1 \
                      + F.l1_loss(volume_ct_reproj_random, image3d) * self.train_cfg.alpha            
        
        im2d_loss_inv = F.l1_loss(figure_ct_reproj_hidden_hidden, figure_ct_source_hidden) * 1 \
                      + F.l1_loss(figure_ct_reproj_hidden_random, figure_ct_source_random) * self.train_cfg.alpha \
                      + F.l1_loss(figure_ct_reproj_random_hidden, figure_ct_source_hidden) * self.train_cfg.alpha \
                      + F.l1_loss(figure_ct_reproj_random_random, figure_ct_source_random) * 1 \
                      
        loss = self.train_cfg.alpha * im2d_loss_inv + self.train_cfg.gamma * im3d_loss_inv  
        
        if self.perc25d_loss is not None:
            perc25d_loss = self.perc25d_loss(volume_ct_reproj_hidden, image3d) \
                         + self.perc25d_loss(volume_ct_reproj_random, image3d) 
            loss += self.train_cfg.lamda * perc25d_loss

        if self.perc30d_loss is not None:
            perc30d_loss = self.perc30d_loss(volume_ct_reproj_hidden, image3d) \
                         + self.perc30d_loss(volume_ct_reproj_random, image3d) 
            loss += self.train_cfg.lamda * perc30d_loss

        self.log(f"{stage}_loss", loss, on_step=(stage == "train"), prog_bar=True, logger=True, sync_dist=True, batch_size=B)
        loss = torch.nan_to_num(loss, nan=1.0) 
        
        # Visualization step
        if batch_idx == 0:
            # Sampling step for X-ray
            with torch.no_grad():
                zeros = torch.zeros_like(image2d)
                
                viz2d = torch.cat([
                    torch.cat([
                        image3d[:, [0], :, self.model_cfg.vol_shape // 2, :], 
                        figure_ct_source_hidden, 
                        figure_ct_source_random, 
                        volume_ct_reproj_hidden[:, [0], :, self.model_cfg.vol_shape // 2, :], 
                        figure_ct_reproj_hidden_hidden, 
                        figure_ct_reproj_hidden_random, 
                    ], dim=-1),
                    torch.cat([
                        image3d[:, [0], :, self.model_cfg.vol_shape // 2, :], 
                        figure_ct_source_hidden, 
                        figure_ct_source_random, 
                        volume_ct_reproj_random[:, [0], :, self.model_cfg.vol_shape // 2, :], 
                        figure_ct_reproj_random_hidden, 
                        figure_ct_reproj_random_random, 
                    ], dim=-1),
                    torch.cat([
                        zeros,
                        figure_xr_source_hidden, 
                        zeros,
                        volume_xr_reproj_hidden[:, [0], :, self.model_cfg.vol_shape // 2, :], 
                        figure_xr_reproj_hidden_hidden, 
                        figure_xr_reproj_hidden_random,
                    ], dim=-1)
                ], dim=-2)

                tensorboard = self.logger.experiment
                grid2d = torchvision.utils.make_grid(viz2d, normalize=False, scale_each=False, nrow=1, padding=0).clamp(0, 1)
                tensorboard.add_image(f"{stage}_df_samples", grid2d, self.current_epoch * B + batch_idx)    
                
        return loss
                        
    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="train")
        self.train_step_outputs.append(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx, stage="validation")
        self.validation_step_outputs.append(loss)
        return loss

    def test_step(self, batch, batch_idx):
        image2d = batch["image2d"]
        image3d = batch["image3d"]
        _device = batch["image3d"].device
        B = image2d.shape[0]

        # Construct the random cameras, -1 and 1 are the same point in azimuths
        dist_hidden = 8 * torch.ones(B, device=_device)
        elev_hidden = torch.zeros(B, device=_device)
        azim_hidden = torch.zeros(B, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=self.model_cfg.fov, znear=self.model_cfg.min_depth, zfar=self.model_cfg.max_depth)

        # Construct the samples in 2D
        figure_xr_source_hidden = image2d
        figure_ct_source_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden)

        # Reconstruct the Encoder-Decoder
        volume_ct_reproj_hidden = self.forward_volume(image2d=figure_ct_source_hidden, cameras=view_hidden)
        psnr = self.psnr(volume_ct_reproj_hidden, image3d)
        ssim = self.ssim(volume_ct_reproj_hidden, image3d)
        self.psnr_outputs.append(psnr)
        self.ssim_outputs.append(ssim)
    
    def on_train_epoch_end(self):
        loss = torch.stack(self.train_step_outputs).mean()
        self.log(
            f"train_loss_epoch",
            loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.train_step_outputs.clear()  # free memory

    def on_validation_epoch_end(self):
        loss = torch.stack(self.validation_step_outputs).mean()
        self.log(
            f"validation_loss_epoch",
            loss,
            on_step=False,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.validation_step_outputs.clear()  # free memory

    def on_test_epoch_end(self):
        print(f"PSNR :{torch.stack(self.psnr_outputs).mean()}")
        print(f"SSIM :{torch.stack(self.ssim_outputs).mean()}")
        self.psnr_outputs.clear()
        self.ssim_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.train_cfg.lr, betas=(0.5, 0.999)
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, #
            milestones=[100, 200, 300, 400], 
            gamma=0.5
        )
        return [optimizer], [scheduler]
