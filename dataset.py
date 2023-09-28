import os
import imageio
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
from geometry_utils import get_rays, pose_spherical

def load_blender_data(basedir, half_res=False, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        if s=='train' or testskip==0:
            skip = 1
        else:
            skip = testskip
            
        for frame in meta['frames'][::skip]:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            imgs.append(imageio.v2.imread(fname))
            poses.append(np.array(frame['transform_matrix']))
        imgs = (np.array(imgs) / 255.).astype(np.float32) # keep all 4 channels (RGBA)
        poses = np.array(poses).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
    
    i_split = [np.arange(counts[i], counts[i+1]) for i in range(3)]
    
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    
    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    if half_res:
        H = H//2
        W = W//2
        focal = focal/2.

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 4))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        # imgs = tf.image.resize_area(imgs, [400, 400]).numpy()
        
    return imgs, poses, render_poses, [H, W, focal], i_split

def sample_stratified(rays_o, rays_d, near, far, n_samples, perturb=True, inverse_depth=False):
    # generate steps between normalized space, this is easier later
    t_vals = torch.linspace(0., 1., n_samples).to(rays_o)

    if not inverse_depth:
        z_vals = near * (1 - t_vals) + far * t_vals
    else:
        # inverse_depth:
        # Sample linearly in inverse depth (disparity)
        z_vals = 1./(1./near * (1.-t_vals) + 1./far * (t_vals))

    # get samples
    if perturb:
        # get the middle points
        mids = .5 * (z_vals[1:] + z_vals[:-1])
        upper = torch.concat([mids, z_vals[-1:]], dim=-1)
        lower = torch.concat([z_vals[:1], mids], dim=-1)
        t_rand = torch.rand([n_samples], device=z_vals.device)
        z_vals = lower + (upper - lower) * t_rand

    z_vals = z_vals.expand(list(rays_o.shape[:-1]) + [n_samples])        

    # Apply scale from `rays_d` and offset from `rays_o` to samples
    # pts: (width, height, n_samples, 3)
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    return pts, z_vals
    

class NeRFData(Dataset):
    def __init__(self, images, poses, focal, near, far, kwargs_sample_stratified, device) -> None:
        super().__init__()
        self.images = images
        self.poses = poses
        self.focal = focal
        self.device = device
        self.near = near
        self.far = far
        self.kwargs_sample_stratified = kwargs_sample_stratified

    def __len__(self):
        return len(self.images)        

    def __getitem__(self, index):
        target_img = self.images[index]
        # target_img = crop_center(target_img)
        height, width = target_img.shape[:2]
        target_pose = self.poses[index]      
        rays_o, rays_d = get_rays(height, width, self.focal, target_pose)
        rays_o = rays_o.reshape([-1, 3])
        rays_d = rays_d.reshape([-1, 3])
        target_img = target_img.reshape([-1, 3])

        query_points, z_vals = sample_stratified(
            rays_o, rays_d, self.near, self.far, **self.kwargs_sample_stratified)        
        
        return {
                'target_pose': target_pose.to(self.device),
                'query_points': query_points.to(self.device), 
                'rays_d': rays_d.to(self.device),
                'z_vals': z_vals.to(self.device),
                'target_image': target_img.to(self.device)}