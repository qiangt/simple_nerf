import tqdm
import numpy as np
import matplotlib.pyplot as plt
import cv2
from typing import List, Callable, Tuple
import imageio

import torch
import torch.nn as nn

from nerf import NeRF
from geometry_utils import get_rays, rot_phi, rot_theta, trans_t, visualize_poses
    

def deg2rad(deg):
    return deg * np.pi / 180.0

def get_chunks(
  inputs: torch.Tensor,
  chunksize: int = 2**15
) -> List[torch.Tensor]:
    r"""
    Divide an input into chunks.
    """
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]

def prepare_chunks(
  points: torch.Tensor,
  chunksize: int = 2**15
) -> List[torch.Tensor]:
    r"""
    Encode and chunkify points to prepare for NeRF model.
    """
    points = points.reshape((-1, 3))
    points = get_chunks(points, chunksize=chunksize)
    return points

def prepare_viewdirs_chunks(
  points: torch.Tensor,
  rays_d: torch.Tensor,
  chunksize: int = 2**15
) -> List[torch.Tensor]:
    r"""
    Encode and chunkify viewdirs to prepare for NeRF model.
    """
    # Prepare the viewdirs
    viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    viewdirs = viewdirs[:, :, None, :].expand(points.shape).reshape((-1, 3))
    viewdirs = get_chunks(viewdirs, chunksize=chunksize)
    return viewdirs

def cumprod_exclusive(
  tensor: torch.Tensor
) -> torch.Tensor:
    r"""
    (Courtesy of https://github.com/krrish94/nerf-pytorch)

    Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
        is to be computed.
    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
        tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """

    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, -1)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, -1)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.
    
    return cumprod


def raw2outputs(
  raw: torch.Tensor,
  z_vals: torch.Tensor,
  rays_d: torch.Tensor,
  raw_noise_std: float = 0.0,
  white_bkgd: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Convert the raw NeRF output into RGB and other maps.
    """

    # Difference between consecutive elements of `z_vals`. [n_rays, n_samples]
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], dim=-1)

    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    # Add noise to model's predictions for density. Can be used to 
    # regularize network during training (prevents floater artifacts).
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[..., 3].shape) * raw_noise_std

    # Predict density of each sample along each ray. Higher values imply
    # higher likelihood of being absorbed at this point. [n_rays, n_samples]
    alpha = 1.0 - torch.exp(-nn.functional.relu(raw[..., 3] + noise) * dists)

    # Compute weight for RGB of each sample along each ray. [n_rays, n_samples]
    # The higher the alpha, the lower subsequent weights are driven.
    weights = alpha * cumprod_exclusive(1. - alpha + 1e-10)

    # Compute weighted RGB map.
    rgb = torch.sigmoid(raw[..., :3])  # [n_rays, n_samples, 3]
    rgb_map = torch.sum(weights[..., None] * rgb, dim=-2)  # [n_rays, 3]

    # Estimated depth map is predicted distance.
    depth_map = torch.sum(weights * z_vals, dim=-1)

    # Disparity map is inverse depth.
    disp_map = 1. / torch.max(1e-10 * torch.ones_like(depth_map),
                                depth_map / torch.sum(weights, -1))

    # Sum of weights along each ray. In [0, 1] up to numerical error.
    acc_map = torch.sum(weights, dim=-1)

    # To composite onto a white background, use the accumulated alpha map.
    if white_bkgd:
        rgb_map = rgb_map + (1. - acc_map[..., None])

    return rgb_map, depth_map, acc_map, weights

def render_rays(model, 
               rays_o,
               rays_d,
               near=2.,
               far=6.,
               N_samples=64,
               chunksize=2**14):
    
    t_vals = torch.linspace(0., 1., N_samples).to(rays_o)
    z_vals = near * (1 - t_vals) + far * t_vals
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]

    batches = prepare_chunks(pts, chunksize=chunksize)

    predictions = []
    for batch  in batches:
        predictions.append(model(batch))    

    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(pts.shape[:3]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)

    return rgb_map, depth_map, acc_map


data = np.load('tiny_nerf_data.npz')
focal = data['focal']
poses = data['poses']

frames = []
H = 100
W = 100
N_samples = 64
c2ws = []

for th in np.linspace(0, 360, 120, endpoint=False):
    # all angles in degrees
    azimuth = th  # rotate along y
    elevation = -30  # rotate along z 
    radius = 4

    c2w = trans_t(radius)
    # based on equation, rotating along x
    c2w = rot_phi(deg2rad(elevation)) @ c2w
    # based on equation, rotation along y
    c2w = rot_theta(deg2rad(azimuth)) @ c2w
    # x => -x, y => z, z => y, after this, it's the opengl coordinate system
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w

    c2ws.append(c2w)

visualize_poses(c2ws)

# for debugging the camera pose
# import sys
# sys.exit(0)

# model parameres and load model
n_freqs = 10
d_input =  3
d_filter = 128
skip = [4]
n_layers = 8
n_freqs_views = 4
log_space = True

nerf_model = NeRF(
    d_input=d_input,
    n_layers=n_layers,
    d_filter=d_filter,
    skip=skip,
    n_freqs=n_freqs,
    log_space=log_space,
    n_freqs_views=n_freqs_views
)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
state_dict = torch.load('nerf_0099.pth')
nerf_model.load_state_dict(state_dict['state_dict'])
nerf_model.eval()
nerf_model.to(device)

# print(nerf_model)

frames = []

for c2w in tqdm.tqdm(c2ws, total=len(c2ws)):
    
    c2w = torch.from_numpy(c2w.astype(np.float32))
    rays_o, rays_d = get_rays(H, W, focal, c2w)

    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)
    rgb, depth, acc = render_rays(nerf_model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)

    rgb_out = rgb.reshape([H, W, 3]).detach().cpu().numpy()
    min_val = np.amin(rgb_out)
    max_val = np.amax(rgb_out)
    rgb_out = 255.0 * (rgb_out - min_val) / (max_val - min_val)
    frames.append(rgb_out.astype(np.uint8))
    
f = 'video.mp4'
imageio.mimwrite(f, frames, fps=30, quality=7)

