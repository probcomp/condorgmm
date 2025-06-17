import condorgmm
import torch
import gsplat
from condorgmm.ng.torch_utils import render_rgbd
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ["CUDA_HOME"] = "/usr/local/cuda"
os.environ["LD_LIBRARY_PATH"] = f"/usr/local/cuda/lib64:{os.environ['LD_LIBRARY_PATH']}"
os.environ["PATH"] = f"/usr/local/cuda/bin:{os.environ['PATH']}"
 
device = torch.device("cuda:0")

condorgmm.rr_init("coarse_models_sweep")

fx,fy,cx,cy = 572.4114, 573.5704, 325.2611, 242.0489
width, height = 640, 480

viewmat = torch.tensor(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    device=device,
    dtype=torch.float32,
)
K = torch.tensor(
    [
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1],
    ],
    device=device,
    dtype=torch.float32,
)

object_mask = np.full((height, width), True)

camera_posquat = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=device, requires_grad=True)
posquat = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], device=device, requires_grad=True)
means = torch.tensor([[0.0, 0.0, 1.0]], device=device, requires_grad=True)
quats = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, requires_grad=True)
scales = torch.tensor([[0.01, 0.01, 0.01]], device=device, requires_grad=True)
opacities = torch.tensor([1.0], device=device, requires_grad=True)
rgbs = torch.tensor([[1.0, 0.0, 0.0]], device=device, requires_grad=True)

print("Rendering")
rendered_rgb, rendered_depth, rendered_silhouette = render_rgbd(
    camera_posquat,
    posquat,
    means,
    quats,
    torch.exp(scales),
    torch.sigmoid(opacities),
    rgbs,
    viewmat[None],
    K[None],
    width,
    height,
)
import matplotlib.pyplot as plt
plt.imshow(rendered_rgb.detach().cpu().numpy())
plt.savefig("test.png")
