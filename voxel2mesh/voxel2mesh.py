import os
import numpy as np
import trimesh
import kaolin


def _voxel2mesh(voxels, threshold=0.5):

    top_verts = [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    top_faces = [[0, 1, 3], [1, 2, 3]]
    top_normals = [[0, 0, 1], [0, 0, 1], [0, 0, 1], [0, 0, 1]]

    bottom_verts = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]]
    bottom_faces = [[1, 0, 3], [2, 1, 3]]
    bottom_normals = [[0, 0, -1], [0, 0, -1], [0, 0, -1], [0, 0, -1]]

    left_verts = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]]
    left_faces = [[0, 1, 3], [2, 0, 3]]
    left_normals = [[-1, 0, 0], [-1, 0, 0], [-1, 0, 0], [-1, 0, 0]]

    right_verts = [[1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    right_faces = [[1, 0, 3], [0, 2, 3]]
    right_normals = [[1, 0, 0], [1, 0, 0], [1, 0, 0], [1, 0, 0]]

    front_verts = [[0, 1, 0], [1, 1, 0], [0, 1, 1], [1, 1, 1]]
    front_faces = [[1, 0, 3], [0, 2, 3]]
    front_normals = [[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]

    back_verts = [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]]
    back_faces = [[0, 1, 3], [2, 0, 3]]
    back_normals = [[0, -1, 0], [0, -1, 0], [0, -1, 0], [0, -1, 0]]

    top_verts = np.array(top_verts)
    top_faces = np.array(top_faces)
    bottom_verts = np.array(bottom_verts)
    bottom_faces = np.array(bottom_faces)
    left_verts = np.array(left_verts)
    left_faces = np.array(left_faces)
    right_verts = np.array(right_verts)
    right_faces = np.array(right_faces)
    front_verts = np.array(front_verts)
    front_faces = np.array(front_faces)
    back_verts = np.array(back_verts)
    back_faces = np.array(back_faces)

    dim = voxels.shape[0]
    new_voxels = np.zeros((dim+2, dim+2, dim+2))
    new_voxels[1:dim+1, 1:dim+1, 1:dim+1] = voxels
    voxels = new_voxels

    scale = 2/dim
    verts = []
    faces = []
    vertex_normals = []
    curr_vert = 0
    a, b, c = np.where(voxels > threshold)

    for i, j, k in zip(a, b, c):
        if voxels[i, j, k+1] < threshold:
            verts.extend(scale * (top_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(top_faces + curr_vert)
            vertex_normals.extend(top_normals)
            curr_vert += len(top_verts)

        if voxels[i, j, k-1] < threshold:
            verts.extend(
                scale * (bottom_verts + np.array([[i-1, j-1, k-1]])))
            faces.extend(bottom_faces + curr_vert)
            vertex_normals.extend(bottom_normals)
            curr_vert += len(bottom_verts)

        if voxels[i-1, j, k] < threshold:
            verts.extend(scale * (left_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(left_faces + curr_vert)
            vertex_normals.extend(left_normals)
            curr_vert += len(left_verts)

        if voxels[i+1, j, k] < threshold:
            verts.extend(scale * (right_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(right_faces + curr_vert)
            vertex_normals.extend(right_normals)
            curr_vert += len(right_verts)

        if voxels[i, j+1, k] < threshold:
            verts.extend(scale * (front_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(front_faces + curr_vert)
            vertex_normals.extend(front_normals)
            curr_vert += len(front_verts)

        if voxels[i, j-1, k] < threshold:
            verts.extend(scale * (back_verts +
                         np.array([[i-1, j-1, k-1]])))
            faces.extend(back_faces + curr_vert)
            vertex_normals.extend(back_normals)
            curr_vert += len(back_verts)

    return np.array(verts) - 1, np.array(faces), np.array(vertex_normals)


def kaolin_voxel2mesh(voxel):
    """
        kaolin install :
            # Replace TORCH_VERSION and CUDA_VERSION with your torch / cuda versions
            pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-{TORCH_VERSION}_cu{CUDA_VERSION}.html
            For example:
                pip install kaolin==0.15.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-1.12.1_cu113.html
    """

    if len(voxel.shape) != 4:
        verts_sk, faces_sk = kaolin.ops.conversions.voxelgrid.voxelgrids_to_cubic_meshes(torch.tensor(voxel).unsqueeze(0))
    else:
        verts_sk, faces_sk = kaolin.ops.conversions.voxelgrid.voxelgrids_to_cubic_meshes(torch.tensor(voxel))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_path", type=str, required=True, default="/public2/home/yuchunan/data/3DVR_draw/deal/shoes/shoe-stl")
    parser.add_argument("--mesh_output", type=str, required=True, default="/public2/home/yuchunan/data/3DVR_draw/deal/shoes/shoe-stl")
    parser.add_argument("--threshold", type=float, required=True, default=0.4)
    args = parser.parse_args()

    suffix = volume_path.split('/')[0].split('.')
    if suffix == 'mat':
        volume = scipy.io.loadmat(volume_path)
        volume = volume['Volume'].astype(np.float32)
    elif suffix == 'binvox':
        with open(volume_path, 'rb') as f:
            volume = utils.binvox_rw.read_as_3d_array(f)
            volume = volume.data.astype(np.float32)


    verts, faces, vertex_normals = _voxel2mesh(volume, threshold)

    if use_vertex_normal:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=vertex_normals)
    else:
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    mesh.exprot(os.path.join(args.mesh_output, 'mesh.obj'))

###############################  kaolin  ###################################################
    # verts, faces = kaolin_voxel2mesh(volume)
    # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    # mesh.exprot(os.path.join(args.mesh_output, 'mesh.obj'))
