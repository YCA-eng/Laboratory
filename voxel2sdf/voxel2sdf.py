import numpy as np
from skimage.measure import marching_cubes
import trimesh

def scale_to_unit_sphere_in_place(mesh):
    assert type(mesh) == trimesh.Trimesh
    mesh.vertices = mesh.vertices - mesh.bounding_box.centroid
    distances = np.linalg.norm(mesh.vertices, axis=1)
    mesh.vertices /= np.max(distances)

def process_sdf(volume, level=0, padding=False, spacing=None, offset=-1, normalize=False, clean=True, smooth=False):
    try:
        if padding:
            volume = np.pad(volume, 1, mode='constant', constant_values=1)
        if spacing is None:
            spacing = 2/volume.shape[-1]
        vertices, faces, normals, _ = marching_cubes(
            volume, level=level, spacing=(spacing, spacing, spacing))
        if offset is not None:
            vertices += offset
        if normalize:
            _mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, vertex_normals=normals)
            scale_to_unit_sphere_in_place(_mesh)
        else:
            _mesh = trimesh.Trimesh(
                vertices=vertices, faces=faces, vertex_normals=normals)
        if clean:
            components = _mesh.split(only_watertight=False)
            bbox = []
            for c in components:
                bbmin = c.vertices.min(0)
                bbmax = c.vertices.max(0)
                bbox.append((bbmax - bbmin).max())
            max_component = np.argmax(bbox)
            _mesh = components[max_component]
        if smooth:
            _mesh = trimesh.smoothing.filter_laplacian(_mesh, lamb=0.05)
        return _mesh
    except Exception as e:
        print(str(e))
        return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_path", type=str, default="/public2/home/yuchunan/data/3DVR_draw/deal/shoes/shoe-stl")
    parser.add_argument("--mesh_output", type=str, default="/public2/home/yuchunan/data/3DVR_draw/deal/shoes/shoe-stl")
    parser.add_argument("--level", type=float, default=0.0)
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--smooth", type=bool, default=False)
    args = parser.parse_args()

    suffix = volume_path.split('/')[0].split('.')
    if suffix == 'mat':
        volume = scipy.io.loadmat(volume_path)
        volume = volume['Volume'].astype(np.float32)
    elif suffix == 'binvox':
        with open(volume_path, 'rb') as f:
            volume = utils.binvox_rw.read_as_3d_array(f)
            volume = volume.data.astype(np.float32)

    mesh = process_sdf(volume, args.level, args.normalize, args.smooth)

    mesh.export(os.path.join(mesh_output, 'mesh.obj'))
