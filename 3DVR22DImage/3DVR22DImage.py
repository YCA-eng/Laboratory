import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse

# 定义视图矩阵和投影矩阵的函数
def look_at(eye, center, up):
    f = np.array(center) - np.array(eye)
    f = f / np.linalg.norm(f)
    up = np.array(up) / np.linalg.norm(up)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)

    M = np.identity(4)
    M[0, :3] = s
    M[1, :3] = u
    M[2, :3] = -f
    M[:3, 3] = -np.array(eye)
    return M

def orthographic_projection(left, right, bottom, top, near, far):
    r_l = right - left
    t_b = top - bottom
    f_n = far - near
    proj = np.array([
        [2.0 / r_l, 0, 0, -(right + left) / r_l],
        [0, 2.0 / t_b, 0, -(top + bottom) / t_b],
        [0, 0, -2.0 / f_n, -(far + near) / f_n],
        [0, 0, 0, 1]
    ])
    return proj

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="./NPY", required=True,help="input folder path")
    parser.add_argument("--output", default="./2D_npy", required=True, help="output folder path")
    args = parser.parse_args()

    # 文件夹路径和输出文件夹
    npy_folder_path = args.input
    output_folder = args.output
    

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 视角配置
    views = {
        'front_view': {'eye': [0, 0, -1], 'center': [0, 0, 0], 'up': [0, 1, 0]},
    }

    # 获取所有.npy文件
    npy_files = [f for f in os.listdir(npy_folder_path) if f.endswith('.npy')]

    # 使用tqdm显示进度条
    for file in tqdm(npy_files, desc="Processing", unit="files"):
        file_path = os.path.join(npy_folder_path, file)
        vertices_3d = np.load(file_path)
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        
        for view_name, params in views.items():
            eye, center, up = params['eye'], params['center'], params['up']
            view_matrix = look_at(eye, center, up)
            proj_matrix = orthographic_projection(-1, 1, -1, 1, 1, 10)
            vp_matrix = np.dot(proj_matrix, view_matrix)
            vertices_2d = np.dot(vp_matrix, np.hstack((vertices_3d, np.ones((vertices_3d.shape[0], 1)))).T)
            vertices_2d = vertices_2d[:2, :] / vertices_2d[3, :]
            
            plt.figure(figsize=(2.24, 2.24), dpi=100)  # Adjust figsize to match 224x224 pixels at 100 dpi
            plt.scatter(vertices_2d[0, :], vertices_2d[1, :], s=1, c='black')  # Set points color to black
            plt.gca().set_aspect('equal', adjustable='box')
            plt.axis('off')
            
            output_file = os.path.join(output_folder, f'{file_name}.png')
            plt.savefig(output_file, dpi=100)  # Ensure the DPI matches the figsize to get 224x224 pixels
            plt.close()
