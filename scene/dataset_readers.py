#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import glob
import cv2
import trimesh

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../permuto_sdf/permuto_sdf_py/models')))
from models import SDF
import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../permuto_sdf/permuto_sdf_py/utils')))
from common_utils import create_bb_for_dataset



class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    print("Liczba kamer: ", len(cam_centers))


    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1
    translate = -center

    print("Center: ", center)
    print("radius: ", radius)
    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []
    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []
    rotation_x_90 = np.array([
                [1, 0, 0, 0],
                [0, 0, -1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1]
            ])

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            c2w = np.dot(rotation_x_90, c2w) 
            c2w[1] *= -1  
            c2w[2] *= -1        
            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            w2c[:3, 3] *= 0.25
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]           

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, model_sdf_path, extension=".png", init_ply_from_sdf = False):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")

    if init_ply_from_sdf:
        aabb = create_bb_for_dataset('nerf')
        sdf = SDF(in_channels=3, boundary_primitive=aabb, geom_feat_size_out=32, nr_iters_for_c2f=10000*1.0).to("cuda")
        sdf.load_state_dict(torch.load(model_sdf_path))
        sdf.eval()
        num_pts = 100_000
        xyz = torch.tensor((np.random.random((num_pts, 3)) - 0.5) * 0.5).float().to("cuda")
        with torch.no_grad():
            sdf_results = sdf(xyz, 200000)[0]
        beta = 1.0
        numerator = torch.exp(beta * sdf_results)
        denominator = (1 + numerator) ** 2
        opacity = ( numerator / denominator  ) * 4
        mask = opacity >= 0.995
        xyz = xyz[mask.squeeze()].cpu().detach().numpy()
        num_pts = xyz.shape[0]

    else:
        num_pts = 100_000
        xyz = (np.random.random((num_pts, 3)) - 0.5) * 0.5
        
    shs = np.random.random((num_pts, 3)) / 255.0
    print(f"Generating random point cloud ({num_pts})...")
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
        
    storePly(ply_path, xyz, SH2RGB(shs) * 255)

    
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def load_K_Rt_from_P(P):

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose


def readCamerasFromDTU(instance_dir, white_background=True, with_mask = True):
    image_dirs = [Path(f"{instance_dir}/images"), Path(f"{instance_dir}/image")]
    image_dir = next((dir for dir in image_dirs if dir.exists()), None)
    if image_dir is None:
        raise ValueError("No image directory.")
    mask_dir = Path(f"{instance_dir}/mask")
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, "*.png")))
    n_images = len(image_paths)
    
    cam_files = [Path(f"{instance_dir}/cameras.npz"), Path(f"{instance_dir}/cameras_sphere.npz")]
    cam_file = next(file for file in cam_files if file.exists())
    camera_dict = np.load(cam_file)
    
    scale_mats = [camera_dict[f'scale_mat_{idx}'].astype(np.float32) for idx in range(n_images)]
    world_mats = [camera_dict[f'world_mat_{idx}'].astype(np.float32) for idx in range(n_images)]


    cam_infos = []
    scene_scale_multiplier = 0.4
    angle = np.deg2rad(115)
    rotation_x_115 = np.array([
            [1, 0, 0, 0],
            [0, np.cos(angle), -np.sin(angle), 0],
            [0, np.sin(angle), np.cos(angle), 0],
            [0, 0, 0, 1]
    ])
    scaling_matrix = np.eye(4)
    scaling_matrix[0, 0] = scene_scale_multiplier
    scaling_matrix[1, 1] = scene_scale_multiplier
    scaling_matrix[2, 2] = scene_scale_multiplier

    for idx, (scale_mat, world_mat) in enumerate(zip(scale_mats, world_mats)):
        image_path = image_paths[idx]
        mask_path = mask_paths[idx]
        image_name = Path(image_path).stem
        image = Image.open(image_path)

        # From 2d gaussian splatting
        P = world_mat @ scale_mat
        P = P[:3, :4]

        intrinsics, pose = load_K_Rt_from_P(P)
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        W = image.width
        H = image.height
        fov_x = 2 * np.arctan(W / (2 * fx)) 
        fov_y = 2 * np.arctan(H / (2 * fy)) 

        c2w = pose
        c2w = np.dot(scaling_matrix, c2w)
        c2w = np.dot(rotation_x_115, c2w)        

        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3] 
        
        if with_mask:
            mask = Image.open(mask_path).convert("L") 
            mask_data = np.array(mask)
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            alpha = norm_data[:, :, 3:4]
            arr = norm_data[:, :, :3] * alpha + bg * (1 - alpha)
            arr[mask_data == 0] = bg
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.uint8), "RGB")
        else:
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=fov_y, FovX=fov_x,
                                     image=image, image_path=image_path,
                                     image_name=image_name, width=image.size[0],
                                     height=image.size[1]))
    return cam_infos


def readDTUSceneInfo(instance_dir, white_background, eval, model_sdf_path, mesh_path):
    print("Reading Dataset DTU Transforms")
    cam_infos = readCamerasFromDTU(instance_dir, white_background)
    train_cam_infos = []
    test_cam_infos = []

    for i, cam in enumerate(cam_infos):
        if (i + 1) % 4 == 0:  
            test_cam_infos.append(cam)
        else:
            train_cam_infos.append(cam)


    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(instance_dir, "points3d.ply")

    aabb = create_bb_for_dataset('dtu')  
    sdf = SDF(in_channels=3, boundary_primitive=aabb, geom_feat_size_out=32, nr_iters_for_c2f=10000 * 1.0).to("cuda")
    sdf.load_state_dict(torch.load(model_sdf_path))
    sdf.eval()


    num_pts = 100_000
    mesh = trimesh.load(mesh_path, force='mesh') 
    points, _ = trimesh.sample.sample_surface(mesh, num_pts)
    xyz = points
    
    xyz = torch.tensor(xyz).float().to("cuda")
    with torch.no_grad():
        sdf_results = sdf(xyz, 200000)[0]
    beta = 300.0
    numerator = torch.exp(beta * sdf_results)
    denominator = (1 + numerator) ** 2
    opacity = (numerator / denominator) * 4
    mask = opacity >= 0.995
    xyz = xyz[mask.squeeze()].cpu().detach().numpy()
    num_pts = xyz.shape[0]
    print("XYZ range: ", xyz.min(), xyz.max())
    
    shs = np.random.random((num_pts, 3)) / 255.0
    print(f"Generating random point cloud ({num_pts})...")
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    
    storePly(ply_path, xyz, SH2RGB(shs) * 255)

    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info   


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "DTU": readDTUSceneInfo
}