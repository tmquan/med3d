from contextlib import contextmanager, nullcontext

import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from kornia.enhance import (
    equalize, 
    equalize3d,
    equalize_clahe 
)

from pytorch3d.renderer.cameras import (
    FoVPerspectiveCameras,
    FoVOrthographicCameras,
    look_at_view_transform,
)
from pytorch3d.renderer.camera_utils import join_cameras_as_batch


from monai.losses import PerceptualLoss
from monai.networks.nets import UNet, VNet, UNETR, SwinUNETR, BasicUNet, ViTAutoEnc, AttentionUnet
from monai.networks.layers.factories import Norm
from monai.utils import optional_import
from monai.metrics import PSNRMetric, SSIMMetric
tqdm, has_tqdm = optional_import("tqdm", name="tqdm")

encoder_feature_channel = {
    "efficientnet-b0": (16, 24, 40, 112, 320),
    "efficientnet-b1": (16, 24, 40, 112, 320),
    "efficientnet-b2": (16, 24, 48, 120, 352),
    "efficientnet-b3": (24, 32, 48, 136, 384),
    "efficientnet-b4": (24, 32, 56, 160, 448),
    "efficientnet-b5": (24, 40, 64, 176, 512),
    "efficientnet-b6": (32, 40, 72, 200, 576),
    "efficientnet-b7": (32, 48, 80, 224, 640),
    "efficientnet-b8": (32, 56, 88, 248, 704),
    "efficientnet-l2": (72, 104, 176, 480, 1376),
}

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

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



def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    # print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


class CustomEmbedding(nn.Module):
    def __init__(self, num_embeddings=4096, embedding_dim=1, v_min=100, v_max=3000):
        super(CustomEmbedding, self).__init__()
        
        # Ensure that 0 < v_min < v_max < num_embeddings
        assert 0 < v_min < v_max < num_embeddings, "Values must satisfy 0 < v_min < v_max < num_embeddings"
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(
            num_embeddings=self.num_embeddings, 
            embedding_dim=self.embedding_dim)
        
        # Initialize the weights using linear spacing
        self.initialize_weights(v_min, v_max)

    def initialize_weights(self, v_min, v_max):
        # Ensure that v_min and v_max are within valid bounds
        assert 0 <= v_min < v_max <= self.num_embeddings, "Invalid range for weight initialization."
        
        # Create a linearly spaced tensor from 0 to 1 with appropriate steps
        steps = v_max - v_min + 1  # Include both endpoints
        weights = torch.linspace(0, 1, steps=steps)
        
        # Assign the initialized weights to the appropriate indices in the lookup table
        self.embedding.weight.data[v_min:v_max + 1] = weights.unsqueeze(1)  # Include v_max
        self.embedding.weight.data[:v_min] = 0.0  # Set weights before v_min to 0.0
        self.embedding.weight.data[v_max + 1:] = 1.0  # Set weights after v_max to 1.0

    def forward(self, indices):
        return self.embedding(indices)
    
class FoVLightningModule(LightningModule):
    def __init__(self, model_cfg: DictConfig, train_cfg: DictConfig):
        super().__init__()

        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        
        # self.lookup_table = nn.Embedding(
        #     num_embeddings=4096, 
        #     embedding_dim=1
        # )
        
        # self.lookup_table.weight.data = torch.linspace(0, 1, steps=4096).unsqueeze(1) 
        # self.lookup_table.weight.data.uniform_(0, 1) #.long()
        # self.lookup_table.weight.data.zero_().long()
        # self.lookup_table = CustomEmbedding(v_min=-512+1024, v_max=3071+1024)
        self.fwd_renderer = ObjectCentricXRayVolumeRenderer(
            image_width=model_cfg.img_shape,
            image_height=model_cfg.img_shape,
            n_pts_per_ray=model_cfg.n_pts_per_ray,
            min_depth=model_cfg.min_depth,
            max_depth=model_cfg.max_depth,
            ndc_extent=model_cfg.ndc_extent,
        )

        self.unet2d_model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=model_cfg.fov_depth,
            num_channels=[256, 256, 512],
            attention_levels=[False, False, True],
            num_head_channels=[0, 0, 512],
            num_res_blocks=2,
            upcast_attention=True,
            use_flash_attention=True,
            dropout_cattn=0.5
            # with_conditioning=True, 
            # cross_attention_dim=12, # Condition with straight/hidden view  # flatR | flatT
        )

        # self.unet2d_model = UNet(
        #     spatial_dims=2,
        #     in_channels=1,
        #     out_channels=model_cfg.fov_depth, 
        #     channels=(64, 128, 256, 512, 1024), #encoder_feature_channel["efficientnet-b8"], 
        #     strides=(2, 2, 2, 2), #(2, 2, 2, 2, 2),
        #     num_res_units=2,
        #     kernel_size=3,
        #     up_kernel_size=3,
        #     act=("LeakyReLU", {"inplace": True}),
        #     dropout=0.5,
        #     # norm=Norm.BATCH,
        #     # mode="pixelshuffle",
        # )

        # self.unet3d_model = DiffusionModelUNet(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=1,
        #     num_channels=[64, 64, 128],
        #     attention_levels=[False, False, True],
        #     num_head_channels=[0, 0, 128],
        #     num_res_blocks=2,
        #     # upcast_attention=True,
        #     # use_flash_attention=True,
        #     # dropout_cattn=0.5
        #     # with_conditioning=True, 
        #     # cross_attention_dim=12, # Condition with straight/hidden view  # flatR | flatT
        # )

        # self.unet2d_model = UNet(
        #     spatial_dims=2,
        #     in_channels=1,
        #     out_channels=model_cfg.fov_depth,
        #     channels=encoder_feature_channel["efficientnet-b8"],
        #     strides=(2, 2, 2, 2),
        #     num_res_units=4,
        #     kernel_size=3,
        #     up_kernel_size=3,
        #     # act=("LeakyReLU", {"inplace": True}),
        #     # dropout=0.5,
        #     # norm=Norm.BATCH,
        # )

        # self.unet3d_model = BasicUNet(
        #     spatial_dims=3, 
        #     in_channels=1, 
        #     out_channels=2, 
        #     features=(32, 32, 64, 128, 256, 32), 
        #     act=('LeakyReLU', {'inplace': True, 'negative_slope': 0.1}), norm=('instance', {'affine': True}), 
        #     bias=True, 
        #     dropout=0.0, 
        #     upsample='pixelshuffle'
        # )

        # self.unet3d_model = AttentionUnet(
        #     spatial_dims=3, 
        #     in_channels=1, 
        #     out_channels=1, 
        #     channels=encoder_feature_channel["efficientnet-b5"], 
        #     strides=(2, 2, 2, 2), 
        #     kernel_size=3, 
        #     up_kernel_size=3, 
        #     # dropout=0.5
        # )

        # self.unet3d_model = None
        self.unet3d_model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=3, 
            channels=(64, 128, 256, 512, 1024), #encoder_feature_channel["efficientnet-b8"], 
            strides=(2, 2, 2, 2), #(2, 2, 2, 2, 2),
            num_res_units=2,
            kernel_size=3,
            up_kernel_size=3,
            act=("LeakyReLU", {"inplace": True}),
            dropout=0.5,
            norm=Norm.BATCH,
            # mode="pixelshuffle",
        )

        # self.unet3d_model = ViTAutoEnc(
        #     in_channels=1, 
        #     patch_size=16,
        #     img_size=model_cfg.vol_shape,
        #     proj_type='conv', 
        #     deconv_chns=16, 
        #     num_layers=8,
        #     num_heads=8,
        # )

        # self.unet3d_model = SwinUNETR(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=1, # value and alpha
        #     img_size=model_cfg.vol_shape, 
        #     feature_size=12,
        #     depths=(2, 2, 2, 2), 
        #     num_heads=(3, 6, 12, 24), 
        #     norm_name='instance', 
        #     drop_rate=0.0, 
        #     attn_drop_rate=0.0, 
        #     dropout_path_rate=0.0, 
        #     normalize=False, 
        #     use_checkpoint=False, 
        #     downsample='mergingv2', 
        #     use_v2=False
        # )

        # self.unet3d_model = UNETR(
        #     spatial_dims=3,
        #     in_channels=1,
        #     out_channels=1, # value and alpha
        #     img_size=model_cfg.vol_shape, 
        #     feature_size=16, 
        #     hidden_size=512, #768, 
        #     mlp_dim=512, #3072, 
        #     num_heads=4, #12, 
        #     proj_type='conv', 
        #     norm_name='instance', 
        #     conv_block=True, 
        #     res_block=True, 
        #     dropout_rate=0.0, 
        #     qkv_bias=False, 
        #     save_attn=False
        # )

        # self.unet3d_model = VNet(
        #     spatial_dims=3, 
        #     in_channels=1,
        #     out_channels=1,
        # )
        
        init_weights(self.unet2d_model, init_type="normal", init_gain=0.02)
        init_weights(self.unet3d_model, init_type="normal", init_gain=0.02)
        
        # self.p20loss = PerceptualLoss(
        #     spatial_dims=2, 
        #     network_type="radimagenet_resnet50", 
        #     is_fake_3d=False, 
        #     pretrained=True,
        # ) 
        
        # self.p25loss = PerceptualLoss(
        #     spatial_dims=3, 
        #     network_type="radimagenet_resnet50", 
        #     is_fake_3d=True, fake_3d_ratio=16/256.,
        #     pretrained=True,
        # ) 

        # self.p30loss = PerceptualLoss(
        #     spatial_dims=3, 
        #     network_type="medicalnet_resnet50_23datasets", 
        #     is_fake_3d=False, 
        #     pretrained=True,
        # ) 

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
    
    def forward_screen(self, image3d, cameras, learnable_windowing=True):
        if learnable_windowing==False:
            image3d = self.correct_window(
                image3d, 
                a_min=-1024, 
                a_max=3071, 
                b_min=-512, 
                b_max=3071
            )
            # image3d = equalize3d(image3d)
        else:
            indices = (4095*image3d.clamp_(0, 1)).long().reshape(image3d.shape[0], 1, -1)
            colored = self.lookup_table(indices)
            image3d = colored.reshape(image3d.shape)
        image2d = self.fwd_renderer(image3d.float(), cameras.float()).clamp_(0, 1)
        # image2d = equalize_clahe(image2d)
        return image2d
        # indices = (4095*image3d.clamp_(0, 1)).long().reshape(B, 1, -1)
        # colored = self.lookup_table(indices)
        # reshape = colored.reshape(image3d.shape)
        # image2d = self.fwd_renderer(reshape, cameras).clamp_(0, 1)
        # return image2d
    
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
        out = self.unet2d_model.forward(image2d, timesteps=timesteps)

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
        elev_random = torch.zeros_like(dist_random, device=_device)
        azim_random = torch.rand_like(dist_random) * 2 - 1  # from [0 1) to [-1 1)
        view_random = make_cameras_dea(dist_random, elev_random, azim_random, fov=self.model_cfg.fov, znear=self.model_cfg.min_depth, zfar=self.model_cfg.max_depth)
    
        dist_hidden = 8 * torch.ones(B, device=_device)
        elev_hidden = torch.zeros(B, device=_device)
        azim_hidden = torch.zeros(B, device=_device)
        view_hidden = make_cameras_dea(dist_hidden, elev_hidden, azim_hidden, fov=self.model_cfg.fov, znear=self.model_cfg.min_depth, zfar=self.model_cfg.max_depth)

        # noise_std_0 = torch.normal(mean=torch.zeros_like(image3d), std=0.05) if stage=='train' else torch.zeros_like(image3d)
        # noise_std_1 = torch.normal(mean=torch.zeros_like(image3d), std=0.05) if stage=='train' else torch.zeros_like(image3d)

        # Construct the samples in 2D
        figure_xr_source_hidden = image2d
        figure_ct_source_hidden = self.forward_screen(image3d=image3d, cameras=view_hidden, learnable_windowing=False)
        figure_ct_source_random = self.forward_screen(image3d=image3d, cameras=view_random, learnable_windowing=False)
            
        # Run the forward pass
        figure_dx_source_concat = torch.cat([figure_xr_source_hidden, figure_ct_source_hidden])
        camera_dx_render_concat = join_cameras_as_batch([view_hidden, view_hidden])

        # For 3D
        volume_dx_reproj_concat = self.forward_volume(
            image2d=figure_dx_source_concat, 
            cameras=camera_dx_render_concat, 
            is_training=(stage=='train')
        )
        volume_xr_reproj_hidden, \
        volume_ct_reproj_hidden = torch.split(volume_dx_reproj_concat, B, dim=0)
           
        figure_xr_reproj_hidden_hidden = self.forward_screen(image3d=volume_xr_reproj_hidden[:,[0],...], cameras=view_hidden, learnable_windowing=False)
        figure_xr_reproj_hidden_random = self.forward_screen(image3d=volume_xr_reproj_hidden[:,[0],...], cameras=view_random, learnable_windowing=False)
        
        figure_ct_reproj_hidden_hidden = self.forward_screen(image3d=volume_ct_reproj_hidden[:,[0],...], cameras=view_hidden, learnable_windowing=False)
        figure_ct_reproj_hidden_random = self.forward_screen(image3d=volume_ct_reproj_hidden[:,[0],...], cameras=view_random, learnable_windowing=False)
        
        im3d_loss_inv = F.l1_loss(volume_ct_reproj_hidden, (image3d)) \
                     
        im2d_loss_inv = F.l1_loss(figure_ct_reproj_hidden_hidden, figure_ct_source_hidden) \
                      + F.l1_loss(figure_ct_reproj_hidden_random, figure_ct_source_random) \
                    
        
        loss = self.train_cfg.alpha * im2d_loss_inv + self.train_cfg.gamma * im3d_loss_inv  
        
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
