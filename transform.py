import torch
import torch.nn.functional as F
import copy
import numpy as np
import trimesh
from os import path
from plyfile import PlyData, PlyElement
from sklearn.neighbors import KDTree
from argparse import ArgumentParser


def load_ts(path):
    return torch.tensor(trimesh.load(path).triangles).to(device)


def save_ply(triangles, path):
    vertices = triangles.reshape(-1, 3).cpu().numpy()
    faces = torch.arange(triangles.numel()).reshape(-1, 3).cpu().numpy()

    vs = np.zeros(vertices.shape[0], dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    vs["x"], vs["y"], vs["z"] = vertices.T

    fs = np.zeros(faces.shape[0], dtype=[("vertex_indices", "i4", (3,))])
    fs["vertex_indices"] = faces

    ply_data = PlyData([PlyElement.describe(vs, "vertex"), PlyElement.describe(fs, "face")])
    ply_data.write(path)


def rot_mat_to_quat(rot: torch.Tensor):
    if rot.size(-1) != 3 or rot.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {rot.shape}.")

    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(rot.reshape(-1, 9), dim=-1)

    q_abs = _sqrt_positive_part(
        torch.stack([
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ], dim=-1))

    quat_by_rijk = torch.stack([
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ], dim=-2)

    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    out = quat_candidates[
          F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
          ].reshape(-1, 4)

    return _standardize_quaternion(out)


def _standardize_quaternion(quaternions):
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _sqrt_positive_part(x):
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def get_basis(triangles, eps=1e-8):
    def norm(u):
        return torch.linalg.vector_norm(u, dim=-1, keepdim=True)

    u0, u1, u2 = triangles.unbind(dim=1)

    _e0 = u1 - u0
    _e1 = u2 - u0
    _e2 = torch.cross(_e0, _e1)

    e0, e1, e2 = _e0 / norm(_e0), _e1 / norm(_e1), _e2 / norm(_e2)
    diag_eps = torch.eye(3, device=triangles.device).unsqueeze(0) * eps

    return torch.stack([e0, e1, e2], dim=2) + diag_eps


def get_degen_ids(mesh, tol=1e-6):
    close = lambda u, v: torch.isclose(u, v, atol=tol).all(dim=1)
    v0, v1, v2 = mesh.unbind(dim=1)
    mask = close(v0, v1) | close(v1, v2) | close(v2, v0)
    return mask.nonzero(as_tuple=True)[0]


def get_transform(mesh, updated):
    degen_ids = get_degen_ids(mesh)
    mesh_centroids = torch.mean(mesh, dim=1).cpu()
    tree = KDTree(mesh_centroids)

    def transform(soup):
        soup_centroids = torch.mean(soup, dim=1).cpu()
        closest_ids_numpy = tree.query(soup_centroids, k=10, return_distance=False)
        closest_ids = torch.from_numpy(closest_ids_numpy).to(device).long()

        first_valid = torch.isin(closest_ids, degen_ids, invert=True).to(torch.int).argmax(dim=1)
        valid_closest = closest_ids[torch.arange(closest_ids.size(0)), first_valid]

        valid_mesh = mesh[valid_closest]
        valid_updated = updated[valid_closest]

        T = get_basis(valid_updated) @ torch.inverse(get_basis(valid_mesh))

        v0, *_ = valid_mesh.unbind(dim=1)
        u0, *_ = valid_updated.unbind(dim=1)

        soup_updated = (soup - v0.unsqueeze(1)) @ T.transpose(1, 2) + u0.unsqueeze(1)

        return soup_updated
    return transform


def recalculate_parameters(triangles):
    def dot(v, u):
        return (v * u).sum(dim=-1, keepdim=True)

    def proj(v, u):
        return dot(v, u) * u

    def norm(u, eps=1e-8):
        return torch.linalg.vector_norm(u, dim=-1, keepdim=True) + eps

    ts = triangles.clone()
    v0, v1, v2 = ts.unbind(dim=1)

    m = v0

    r0 = torch.cross(v1 - v0, v2 - v0)
    r0 /= norm(r0)

    r1 = v1 - v0
    r1 /= norm(r1)

    r2 = v2 - v0 - proj(v2 - v0, r0) - proj(v2 - v0, r1)
    r2 /= norm(r2)

    r = torch.stack([r0, r1, r2], dim=2)

    s1 = norm(v1 - v0)
    s2 = dot(v2 - v0, r2)

    s = torch.cat([s1, s2], dim=1)

    return m, F.normalize(rot_mat_to_quat(r)), torch.log(s)

def get_param(g, param_name):
    params = torch.stack([torch.Tensor(g[d]) for d in dims[param_name]], dim=-1)
    return F.normalize(params) if param_name == "rotation" else params


def update_gaussians(g, updated_params):
    enum_dims = lambda dims: zip(dims, range(len(dims)))
    _g = copy.deepcopy(g)

    for param_name, updated_param in updated_params.items():
        for d, id in enum_dims(dims[param_name]):
            _g[d] = updated_param[:, id].cpu()

    return _g


def compare(g, updated_params):
    for param_name, updated_param in updated_params.items():
        max_absolute_diff = (get_param(g, param_name) - updated_param.cpu()).abs().max()
        print(f"{param_name}: {max_absolute_diff}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Transform script parameters")
    parser.add_argument("--source_dir", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--triangle_soup_name", type=str, default="30000.obj")
    parser.add_argument("--mesh_original_name", type=str)
    parser.add_argument("--mesh_modified_name", type=str)
    parser.add_argument("--gaussians_name", type=str, default="point_cloud.ply")
    args = parser.parse_args()
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source_dir = args.source_dir
    output_dir = args.output_dir

    source_filenames = {
        "triangle_soup": args.triangle_soup_name,
        "mesh_original": args.mesh_original_name,
        "mesh_modified": args.mesh_modified_name,
        "gaussians": args.gaussians_name
    }

    output_filenames = {
        "triangle_soup": f"modified_triangles.ply",
        "gaussians": f"modified_gaussians.ply"
    }

    src_paths =  {
        key: path.join(source_dir, filename)
        for key, filename in source_filenames.items()
    }

    out_paths = {
        key: path.join(output_dir, filename)
        for key, filename in output_filenames.items()
    }


    datasets = ["triangle_soup", "mesh_original", "mesh_modified"]
    soup, mesh_original, mesh_modified = (load_ts(src_paths[name]) for name in datasets)

    transform = get_transform(mesh_original, mesh_modified)
    soup_modified = transform(soup)
    save_ply(soup_modified, out_paths["triangle_soup"])

    xyz_dims = ("x", "y", "z")
    rot_dims = ("rot_0", "rot_1", "rot_2", "rot_3")
    scl_dims = ("scale_1", "scale_2")

    params = ("xyz", "rotation", "scaling")
    dims = dict(zip(params, (xyz_dims, rot_dims, scl_dims)))

    recalculated_params = dict(zip(params, recalculate_parameters(soup_modified)))
    gaussians = PlyData.read(src_paths["gaussians"])["vertex"]
    updated_gaussians = update_gaussians(gaussians, recalculated_params)
    PlyData([updated_gaussians]).write(out_paths["gaussians"])