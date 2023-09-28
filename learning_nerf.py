import os
import imageio
import json
import matplotlib.pyplot as plt
import numpy as np
import cv2
from typing import Dict,Tuple,Optional,List,Callable
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nerf import NeRF
from dataset import NeRFData

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

def sample_pdf(
  bins: torch.Tensor,
  weights: torch.Tensor,
  n_samples: int,
  perturb: bool = False
) -> torch.Tensor:
    r"""
    Apply inverse transform sampling to a weighted set of points.
    """
    # Normalize weights to get PDF.
    pdf = (weights + 1e-5) / torch.sum(weights + 1e-5, -1, keepdims=True) # [n_rays, weights.shape[-1]]

    # Convert PDF to CDF.
    cdf = torch.cumsum(pdf, dim=-1) # [n_rays, weights.shape[-1]]
    cdf = torch.concat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1) # [n_rays, weights.shape[-1] + 1]

    # Take sample positions to grab from CDF. Linear when perturb == 0.
    if not perturb:
        u = torch.linspace(0., 1., n_samples, device=cdf.device)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples]) # [n_rays, n_samples]
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples], device=cdf.device) # [n_rays, n_samples]

    # Find indices along CDF where values in u would be placed.
    u = u.contiguous() # Returns contiguous tensor with same values.
    inds = torch.searchsorted(cdf, u, right=True) # [n_rays, n_samples]

    # Clamp indices that are out of bounds.
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], dim=-1) # [n_rays, n_samples, 2]

    # Sample from cdf and the corresponding bin centers.
    matched_shape = list(inds_g.shape[:-1]) + [cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(-2).expand(matched_shape), dim=-1,
                        index=inds_g)
    bins_g = torch.gather(bins.unsqueeze(-2).expand(matched_shape), dim=-1,
                            index=inds_g)

    # Convert samples to ray length.
    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples # [n_rays, n_samples]

def sample_hierarchical(
  rays_o: torch.Tensor,
  rays_d: torch.Tensor,
  z_vals: torch.Tensor,
  weights: torch.Tensor,
  n_samples: int,
  perturb: bool = False
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""
    Apply hierarchical sampling to the rays.
    """

    # Draw samples from PDF using z_vals as bins and weights as probabilities.
    z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    new_z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1], n_samples,
                            perturb=perturb)
    new_z_samples = new_z_samples.detach()

    # Resample points from ray based on PDF.
    z_vals_combined, _ = torch.sort(torch.cat([z_vals, new_z_samples], dim=-1), dim=-1)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals_combined[..., :, None]  # [N_rays, N_samples + n_samples, 3]
    return pts, z_vals_combined, new_z_samples

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

def nerf_forward(
        data,
        chunksize: int,
        coarse_model: nn.Module,
        kwargs_sample_stratified: dict=None,
        kwargs_sample_hierarchical: dict=None
):
    # Set no kwargs if none are given.
    if kwargs_sample_stratified is None:
        kwargs_sample_stratified = {}
    if kwargs_sample_hierarchical is None:
        kwargs_sample_hierarchical = {}    

    query_points = data['query_points']
    rays_d = data['rays_d']
    z_vals = data['z_vals']

    batches = prepare_chunks(query_points, chunksize=chunksize)

    # Coarse model pass.
    # Split the encoded points into "chunks", run the model on all chunks, and
    # concatenate the results (to avoid out-of-memory issues).
    # query point is [b, n_rays, n_samples, 3]
    predictions = []
    for batch in batches:
        predictions.append(coarse_model(batch))

    raw = torch.cat(predictions, dim=0)
    raw = raw.reshape(list(query_points.shape[1:3]) + [raw.shape[-1]])

    # Perform differentiable volume rendering to re-synthesize the RGB image.
    rgb_map, depth_map, acc_map, weights = raw2outputs(raw, z_vals, rays_d)
    # rgb_map, depth_map, acc_map, weights = render_volume_density(raw, rays_o, z_vals)
    outputs = {
        'z_vals_stratified': z_vals
    }    
        
    # Store outputs.
    outputs['rgb_map'] = rgb_map
    outputs['depth_map'] = depth_map
    outputs['acc_map'] = acc_map
    outputs['weights'] = weights
    return outputs        

def plot_samples(
    z_vals: torch.Tensor,
    z_hierarch: Optional[torch.Tensor] = None,
    ax: Optional[np.ndarray] = None):
    r"""
    Plot stratified and (optional) hierarchical samples.
    """
    y_vals = 1 + np.zeros_like(z_vals)

    if ax is None:
        ax = plt.subplot()
    ax.plot(z_vals, y_vals, 'b-o')
    if z_hierarch is not None:
        y_hierarch = np.zeros_like(z_hierarch)
        ax.plot(z_hierarch, y_hierarch, 'r-o')
    ax.set_ylim([-1, 2])
    ax.set_title('Stratified  Samples (blue) and Hierarchical Samples (red)')
    ax.axes.yaxis.set_visible(False)
    ax.grid(True)
    return ax

def crop_center(
  img: torch.Tensor,
  frac: float = 0.5
) -> torch.Tensor:
    r"""
    Crop center square from image.
    """
    h_offset = round(img.shape[0] * (frac / 2))
    w_offset = round(img.shape[1] * (frac / 2))
    return img[h_offset:-h_offset, w_offset:-w_offset]
    
def init_models(
        d_input, 
        n_freqs, 
        log_space, 
        n_freqs_views, 
        n_layers, 
        d_filter, 
        skip, 
        device='cpu'):
    
    model = NeRF(d_input=d_input, n_layers=n_layers, d_filter=d_filter, skip=skip,
                 n_freqs=n_freqs, log_space=log_space, n_freqs_views=n_freqs_views)
    model.to(device)
    model_params = list(model.parameters())    
  
    # Optimizer
    optimizer = torch.optim.Adam(model_params, lr=lr)

    return model, optimizer    


# datadir = 'data/nerf_synthetic/lego'
# visualize = False
# images, poses, render_poses, hwf, i_split = load_blender_data(datadir)

data = np.load('tiny_nerf_data.npz')
images = data['images']
poses = data['poses']
focal = data['focal']

height, width = images.shape[1:3]
near, far = 2., 6.

n_training = 100
testimg_idx = 101
testimg, testpose = images[testimg_idx], poses[testimg_idx]

plt.imshow(testimg)
print('Pose')
print(testpose)

# Gather as torch tensors
device = 'cuda:0' if torch.cuda.is_available else 'cpu'
images = torch.from_numpy(images)
poses = torch.from_numpy(poses)

# try the real model now
# first define a bunch of model paramers
# Encoders
d_input = 3           # Number of input dimensions
n_freqs = 10          # Number of encoding functions for samples
log_space = True      # If set, frequencies scale in log space
n_freqs_views = 4     # Number of encoding functions for views

# Stratified sampling
n_samples = 64         # Number of spatial samples per ray
perturb = True         # If set, applies noise to sample positions
inverse_depth = False  # If set, samples points linearly in inverse depth

# Model
d_filter = 128          # Dimensions of linear layer filters
n_layers = 8            # Number of layers in network bottleneck
skip = [4]               # Layers at which to apply input residual

# Hierarchical sampling
n_samples_hierarchical = 64   # Number of samples per ray
perturb_hierarchical = False  # If set, applies noise to sample positions

# Optimizer
lr = 5e-4  # Learning rate

# Training
batch_size = 2**14          # Number of rays per gradient step (power of 2)
display_rate = 25          # Display test output every X epochs
chunksize = 2**14

# Early Stopping
warmup_iters = 100          # Number of iterations during warmup phase
warmup_min_fitness = 10.0   # Min val PSNR to continue training at warmup_iters

# We bundle the kwargs for various functions to pass all at once.
kwargs_sample_stratified = {
    'n_samples': n_samples,
    'perturb': perturb,
    'inverse_depth': inverse_depth
}
kwargs_sample_hierarchical = {
    'perturb': perturb
}


model, optimizer = init_models(
    d_input, 
    n_freqs,
    log_space,
    n_freqs_views,
    n_layers,
    d_filter,
    skip,
    device)

train_psnrs = []
val_psnrs = []
iternums = []

# height, width = images.shape[1:3]
# height = round(height * keep_ratio)
# width = round(width * keep_ratio)

nerf_train_data = NeRFData(images[:n_training, ...], 
                     poses[:n_training, ...], 
                     focal,
                     near,
                     far,
                     kwargs_sample_stratified,
                     device)

nerf_test_data = NeRFData(images[n_training:, ...], 
                     poses[n_training:, ...], 
                     focal,
                     near,
                     far,
                     kwargs_sample_stratified,
                     device)

train_dataloader = DataLoader(nerf_train_data, shuffle=True)
test_dataloader = DataLoader(nerf_test_data)
n_epoch = 100

for epoch in range(n_epoch):
    print('Training epoch {}'.format(epoch))
    for i, data in tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        outputs = nerf_forward(
            data,
            chunksize,
            coarse_model=model,
            kwargs_sample_stratified=kwargs_sample_stratified,
            kwargs_sample_hierarchical=kwargs_sample_hierarchical,
        )

        # Check for any numerical issues.
        for k, v in outputs.items():
            if torch.isnan(v).any():
                print(f"! [Numerical Alert] {k} contains NaN.")
            if torch.isinf(v).any():
                print(f"! [Numerical Alert] {k} contains Inf.")

        # Backprop!
        rgb_predicted = outputs['rgb_map']
        loss = torch.nn.functional.mse_loss(rgb_predicted, data['target_image'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        psnr = -10. * torch.log10(loss)
        train_psnrs.append(psnr.item())

        # Evaluate testimg at given display rate.

        if i % display_rate == 0:
            test_data = next(iter(test_dataloader))
            model.eval()
            outputs = nerf_forward(
                test_data,
                chunksize,
                coarse_model=model,
                kwargs_sample_stratified=kwargs_sample_stratified,
                kwargs_sample_hierarchical=kwargs_sample_hierarchical,
            )
            rgb_predicted = outputs['rgb_map']
            loss = torch.nn.functional.mse_loss(rgb_predicted, test_data['target_image'].reshape(-1, 3))
            print("Loss:", loss.item())
            val_psnr = -10. * torch.log10(loss)
            val_psnrs.append(val_psnr.item())
            iternums.append(epoch * len(train_dataloader) + i)

            # Plot example outputs
            fig, ax = plt.subplots(1, 4, figsize=(20,6), gridspec_kw={'width_ratios': [1, 1, 1, 3]})
            ax[0].imshow(rgb_predicted.reshape([height, width, 3]).detach().cpu().numpy())
            ax[0].set_title(f'Epoch: {epoch} Iteration: {i}')
            ax[1].imshow(test_data['target_image'].reshape([height, width, 3]).detach().cpu().numpy())
            ax[1].set_title(f'Test Target')
            ax[2].plot(range(0, epoch * len(train_dataloader) + i + 1), train_psnrs, 'r')
            ax[2].plot(iternums, val_psnrs, 'b')
            ax[2].set_title('PSNR (train=red, val=blue')
            z_vals_strat = outputs['z_vals_stratified'].view((-1, n_samples))
            z_sample_strat = z_vals_strat[z_vals_strat.shape[0] // 2].detach().cpu().numpy()
            if 'z_vals_hierarchical' in outputs:
                z_vals_hierarch = outputs['z_vals_hierarchical'].view((-1, n_samples_hierarchical))
                z_sample_hierarch = z_vals_hierarch[z_vals_hierarch.shape[0] // 2].detach().cpu().numpy()
            else:
                z_sample_hierarch = None
            _ = plot_samples(z_sample_strat, z_sample_hierarch, ax=ax[3])
            ax[3].margins(0)
            # plt.show()
            plt.savefig('test.jpg')

        # save an model every 50 epoch
        # save model
        if psnr > 30.0:
            # good enough model
            state = {'state_dict': model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'loss': loss}

            torch.save(state, 'nerf_{:04d}.pth'.format(epoch))        
            

print('')
print(f'Done!')

# always saving the last model
state = {'state_dict': model.state_dict(),
        'optimizer':  optimizer.state_dict(),
        'loss': loss}

torch.save(state, 'nerf_{:04d}.pth'.format(epoch))       



