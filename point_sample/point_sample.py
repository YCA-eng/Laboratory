import os
import scipy
import torch
from scipy.io import loadmat
import numpy as np
import trimesh
import skimage
import argparse

from pytorch3d.loss import chamfer_distance
import skimage.measure
from pytorch3d.io import load_ply
from pytorch3d.io import load_obj
from pytorch3d.io import save_obj
import plyfile
import logging

from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.ops import sample_points_from_meshes

import tqdm
import open3d as o3d

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def normalize_to_box(input):
    """
    normalize point cloud to unit bounding box
    center = (max - min)/2
    scale = max(abs(x))
    input: pc [N, P, dim] or [P, dim]
    output: pc, centroid, furthest_distance

    From https://github.com/yifita/pytorch_points
    """
    if len(input.shape) == 2:
        axis = 0
        P = input.shape[0]
        D = input.shape[1]
    elif len(input.shape) == 3:
        axis = 1
        P = input.shape[1]
        D = input.shape[2]
    else:
        raise ValueError()

    if isinstance(input, np.ndarray):
        maxP = np.amax(input, axis=axis, keepdims=True)
        minP = np.amin(input, axis=axis, keepdims=True)
        centroid = (maxP + minP) / 2
        input = input - centroid
        furthest_distance = np.amax(np.abs(input), axis=(axis, -1), keepdims=True)
        input = input / furthest_distance
    elif isinstance(input, torch.Tensor):
        maxP = torch.max(input, dim=axis, keepdim=True)[0]
        minP = torch.min(input, dim=axis, keepdim=True)[0]
        centroid = (maxP + minP) / 2
        input = input - centroid
        in_shape = list(input.shape[:axis]) + [P * D]
        furthest_distance = torch.max(torch.abs(input).reshape(in_shape), dim=axis, keepdim=True)[0]
        furthest_distance = furthest_distance.unsqueeze(-1)
        input = input / furthest_distance
    else:
        raise ValueError()

    return input, centroid, furthest_distance

def save_point_cloud_to_obj(vertices, file_path):
    """
    保存点云数据到 OBJ 文件

    参数：
    - vertices：点云的顶点坐标，形状为 (N, 3)，N 表示点的数量，每个点由三个坐标组成。
    - file_path：要保存的 OBJ 文件路径。
    """
    with open(file_path, 'w') as f:
        for vertex in vertices:
            f.write(f'v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n')

def sample_obj(obj_paths):
    mesh_list = []
    for obj_path in obj_paths:
        if not os.path.exists(obj_path):
            continue
        verts, faces, _ = load_obj(obj_path)
        mesh = Meshes(verts=[verts], faces=[faces.verts_idx])
        mesh_list.append(mesh)
    meshes = join_meshes_as_batch(mesh_list)
    pcs = sample_points_from_meshes(
                meshes, num_samples=4096)
    return pcs

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input", default='demo.obj',required=True, help="input file path")
    # parser.add_argument("--output",default='.', required=True, help="output file path")
    parser.add_argument("--input", default='demo.obj', help="input file path")
    parser.add_argument("--output",default='.', help="output file path")
    args = parser.parse_args()

    output_directory = os.path.dirname(args.output)
    ensure_directory(output_directory)
    
    dp_pc = sample_obj([args.input])[0]
    dp_pc = normalize_to_box(dp_pc)[0]
    save_point_cloud_to_obj(dp_pc.cpu().numpy(), args.output)