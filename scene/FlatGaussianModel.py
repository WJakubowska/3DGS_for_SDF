
#
# Copyright (C) 2024, Gmum
# Group of Machine Learning Research. https://gmum.net/
# All rights reserved.
#
# The Gaussian-splatting software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr
#
# The Gaussian-mesh-splatting is software based on Gaussian-splatting, used on research.
# This Games software is free for non-commercial, research and evaluation use


import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, build_rotation
from torch import nn
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from scene.gaussian_model import GaussianModel

class FlatGaussianModel(GaussianModel):

    def __init__(self, sh_degree: int, model_sdf_path: str, beta: float):

        super().__init__(sh_degree, model_sdf_path, beta)
        self.eps_s0 = 1e-8
        self.s0 = torch.empty(0)
        self.log_eps_s0 = torch.log(torch.tensor(self.eps_s0))

    @property
    def get_scaling(self):
        self.s0 = torch.ones(self._scaling.shape[0], 1).cuda() * self.eps_s0
        return torch.cat([self.s0, self.scaling_activation(self._scaling[:, [-2, -1]])], dim=1)
    
    @property
    def get_scaling_without_activation(self):
        _s0 = torch.ones(self._scaling.shape[0], 1).cuda() * self.log_eps_s0
        return torch.cat([_s0, self._scaling[:, [-2, -1]]], dim=1)

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        opacities = inverse_sigmoid(self.get_opacity)
        self._opacity = opacities
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent
        )
        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8*N)
        )[:, [1, 2]]
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self.get_new_opacities(new_xyz).repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def prepare_triangles(self):
        """
        Prepare psudo-mesh face based on Gaussian.
        """
        scales = self.get_scaling
        rotation = self._rotation
        R = build_rotation(rotation)
        R = R.transpose(-2, -1)

        v1 = self._xyz
        s_2 = scales[:, -2]
        s_3 = scales[:, -1]
        _v2 = v1 + s_2.reshape(-1, 1) * R[:, 1]
        _v3 = v1 + s_3.reshape(-1, 1) * R[:, 2]

        mask = s_2 > s_3

        v2 = torch.zeros_like(_v2)
        v3 = torch.zeros_like(_v3)

        v2[mask] = _v2[mask]
        v3[mask] = _v3[mask]

        v2[~mask] = _v3[~mask]
        v3[~mask] = _v2[~mask]

        v1 = v1
        v2 = v2
        v3 = v3

        triangles = torch.stack([v1, v2, v3], dim = 1)
        return triangles
    
    def save_flat_faces(self, filename):
        triangles =  self.prepare_triangles()
        vertices = triangles.reshape(-1, 3)  
        vertices_text = "\n".join(f"v {x} {y} {z}" for x, y, z in vertices)

        num_triangles = triangles.shape[0]
        faces_indices = np.arange(1, 3 * num_triangles + 1).reshape(-1, 3)
        faces_text = "\n".join(f"f {a} {b} {c}" for a, b, c in faces_indices)
        
        with open(filename, 'w') as f:
            f.write(vertices_text + "\n" + faces_text)
        print(f"Saving completed. Faces available at {filename}")
    
    def calculate_loss_from_sdf(self):
        triangles =  self.prepare_triangles()
        v1 = triangles[:, 0]
        gaussian_center = v1
        num_gaussian_center = v1.shape[0]
        with torch.no_grad():
            sdf_results = self.sdf(gaussian_center, 200000)[0]
        centroids_sum = torch.sum(torch.abs(sdf_results))
        loss = 1 / num_gaussian_center * centroids_sum
        return loss
