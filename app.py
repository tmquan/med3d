import torch
import torchvision
import gradio as gr

from tqdm import tqdm
import numpy as np
import cv2
from monai.transforms import (
    Compose, 
    LoadImageDict, 
    EnsureChannelFirstDict, 
    ScaleIntensityDict, 
    HistogramNormalizeDict,
    CropForegroundDict, 
    ZoomDict, 
    ResizeDict, 
    DivisiblePadDict, 
    ToTensorDict
)

from nvlitmodel import NVLightningModule, make_cameras_dea

# Load the model
# chkpt = "logs/diffusion/version_0/checkpoints/epoch=79-step=80000.ckpt"
chkpt = "logs/diffusion/version_0/checkpoints/last.ckpt"

B = 1
device = torch.device('cuda:0')
# model = NVLightningModule(model_cfg=None, train_cfg=None)  # Initialize your model
# model.load_state_dict(torch.load(chkpt))  # Load model weights
model = NVLightningModule.load_from_checkpoint(chkpt)
model = model.to(device)
model.eval()  # Set model to evaluation mode

transforms = Compose(
    [
        LoadImageDict(keys=["image2d"]),
        EnsureChannelFirstDict(keys=["image2d"],),
        ScaleIntensityDict(keys=["image2d"], minv=0.0, maxv=1.0),
        HistogramNormalizeDict(keys=["image2d"], min=0.0, max=1.0),
        CropForegroundDict(keys=["image2d"], source_key="image2d", select_fn=lambda x: x > 0, margin=0),
        ZoomDict(keys=["image2d"], zoom=0.95, padding_mode="constant", mode="area"),
        ResizeDict(keys=["image2d"], spatial_size=256, size_mode="longest", mode="area"),
        DivisiblePadDict(keys=["image2d"], k=(256, 256), mode="constant", constant_values=0),
        ToTensorDict(keys=["image2d"], )
    ]
)

def predict(input2d):
    cv2.imwrite("demo/sample.png", input2d)
    input2d_dict = {"image2d": "demo/sample.png"}
    input2d_tfms = transforms(input2d_dict)
    image2d = input2d_tfms["image2d"]
    image2d = image2d.unsqueeze(0)  # Add batch dimension
    image2d = image2d.to(device)
    # print(image2d.shape)

    # Generate the volume
    dist = 8 * torch.ones(B, device=device)
    elev = torch.zeros(B, device=device)
    azim = torch.zeros(B, device=device)
    cameras = make_cameras_dea(dist, elev, azim, fov=16.0, znear=6.1, zfar=9.9)
    cameras = cameras.to(device)

    with torch.no_grad():
        volume = model.forward_volume(
            image2d=image2d, 
            cameras=cameras, 
        ).clamp(0.0, 1.0) # volume = torch.permute(volume, (0, 1, 3, 4, 2)) inside
        
    # Generate the rotating screen
    frames = []
    for az in tqdm(range(256)):
        azim = (az / 256 * 2) * torch.ones(B, device=device)
        cameras = make_cameras_dea(dist, elev, azim, fov=16.0, znear=6.1, zfar=9.9)
        cameras.to(device)
        
        screen = model.forward_screen(
            image3d=volume, 
            cameras=cameras, 
        ).squeeze().detach().cpu() 
        
        # screen = torch.flip(screen, (1,))
        # screen = torch.flip(screen, (2,))
        screen = screen.transpose(1, 0) # Tensorboard
        frames.append(screen)

    frames = torch.from_numpy(np.array(frames))
    # return frames

    rotates = frames.squeeze().detach().cpu() #.numpy()

    # # Determine the midpoint
    # midpoint = len(rotates) // 4
    # rotates = torch.cat([rotates[midpoint:,:,:], rotates[:midpoint,:,:]], dim=0)

    # Convert the input image to a video
    height, width = (256, 256)
    video_name = '360view.mp4'
    rotates = 255*rotates.unsqueeze(-1).repeat(1,1,1,3)
    volume = 255*volume.squeeze().detach().cpu().unsqueeze(-1).repeat(1,1,1,3)
    
    torchvision.io.write_video(
        filename = video_name,
        video_array = rotates,
        fps = 32.0
    )

    # concat = torch.cat([
    #     torch.cat([rotates, volume.transpose(1, 2)], dim=1),
    #     torch.cat([volume.transpose(1, 2).transpose(0, 1), volume.transpose(1, 2).transpose(0, 2)], dim=1),
    # ], dim=2)
    # torchvision.io.write_video(
    #     filename = video_name,
    #     video_array = concat,
    #     fps = 32.0
    # )
    
    return video_name
       

# Create Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(
        type="numpy", 
        image_mode="L", 
        height=256),
    examples=[
        # "data/ChestXRLungSegmentation/VinDr/v1/processed/test/images/0a8dc3de200bc767169b73a6ab91e948.png",
        # "data/ChestXRLungSegmentation/VinDr/v1/processed/test/images/0a2e45455fe2c76ddbbb5f38ab35f6f2.png",
        # "data/ChestXRLungSegmentation/VinDr/v1/processed/test/images/0a6fd1c1d71ff6f9e0f0afa746e223e4.png",
        # "data/ChestXRLungSegmentation/VinDr/v1/processed/test/images/0b2cc81ad04ca2e91f2a8626b645cad8.png",
        # "data/ChestXRLungSegmentation/VinDr/v1/processed/test/images/1b4407d5e1e9ec1e602c75fc80b7b001.png",
        "demo/xr/0ccde8751e5952ea6bb047378dcbabba.png",
        "demo/xr/0c15e19c74ef8ddd9bed0a4fc7f4b5a8.png",
        "demo/xr/002a34c58c5b758217ed1f584ccbcfe9.png",
        "demo/xr/3ba1428b5a468133d0b75b0418514c23.png",
        "demo/xr/4aace7806718a917535d92c97eec097c.png",
        "demo/xr/4bbddf35e1c4dfaa9c29d1fca909575f.png",
        "demo/xr/4d2071baccfd994701c3a5ceb3f64f6f.png",
        "demo/xr/4fef5faf22105d6c29f218236dd44d66.png",
        # "demo/xr/0cd194f171015c0eb824a1bd057f440b.png",
        # "demo/xr/0ce1b656d171e2e760340f29f4a064e9.png",
        # "demo/xr/0ce3ae82d1f5c69f253432c5ec47c1be.png",
        # "demo/xr/0d0561a5e6379aba13d71e50b6b70747.png",
        # "demo/xr/12d598be7970fc60aa76bd7c21eef405.png",
        # "demo/xr/12df63c7ee868a1677ba273c3c775476.png",
        # "demo/xr/12e58d45f46e91b639072bd5a86bea3e.png",
        # "demo/ct/data_ChestXRLungSegmentation_MOSMED_processed_train_images_CT-0_study_0001.png",
        # "demo/ct/data_ChestXRLungSegmentation_MOSMED_processed_train_images_CT-0_study_0002.png",
        # "demo/ct/data_ChestXRLungSegmentation_MOSMED_processed_train_images_CT-0_study_0003.png",
        # "demo/ct/data_ChestXRLungSegmentation_MOSMED_processed_train_images_CT-0_study_0004.png",
        # "demo/ct/data_ChestXRLungSegmentation_MOSMED_processed_train_images_CT-0_study_0005.png",
        # "demo/ct/data_ChestXRLungSegmentation_MOSMED_processed_train_images_CT-0_study_0006.png",
        # "demo/ct/data_ChestXRLungSegmentation_MOSMED_processed_train_images_CT-0_study_0007.png",
        # "demo/ct/data_ChestXRLungSegmentation_MOSMED_processed_train_images_CT-0_study_0008.png",
    ],
    outputs=gr.Video(
        height=256,
    )
)

# Launch the Gradio app
if __name__ == "__main__":
    iface.launch(share=True)
