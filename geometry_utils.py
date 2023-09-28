import torch
import numpy as np
import matplotlib.pyplot as plt

trans_t = lambda t : np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]], dtype=np.float32)

rot_phi = lambda phi : np.array([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]], dtype=np.float32)

rot_theta = lambda th : np.array([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]], dtype=np.float32)

def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

def visualize_poses(poses):
    dirs = np.stack([np.sum([np.sum([0, 0, -1] * pose[:3, :3], axis=-1)]) for pose in poses])
    origins = np.stack([pose[:3, -1] for pose in poses])
    ax = plt.figure(figsize=(12, 8)).add_subplot(projection='3d')
    _ = ax.quiver(
    origins[..., 0].flatten(),
    origins[..., 1].flatten(),
    origins[..., 2].flatten(),
    dirs[..., 0].flatten(),
    dirs[..., 1].flatten(),
    dirs[..., 2].flatten(),
)
    ax.set_xlabel('X')
    ax.set_xlabel('Y')
    ax.set_xlabel('Z')
    plt.show()


def get_rays(height, width, focal, c2w):
    i, j = torch.meshgrid(
         torch.arange(width, dtype=torch.float32).to(c2w),
         torch.arange(height, dtype=torch.float32).to(c2w),
         indexing='ij')
    
    i, j = i.transpose(1, 0), j.transpose(1, 0)
    # inverse of intrinsic to unproject the pixel coordinate to world
    directions = torch.stack(
        [(i - width*0.5)/focal,
         -(j - height*0.5)/focal,
         -torch.ones_like(i)], dim=-1)
    # applying transform matrix
    rays_d = torch.sum(directions[:, :, None, :] * c2w[:3, :3], dim=-1)

    # get origin
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d 