'''
    STL to OBJ
'''

import numpy as np
from stl import mesh
from tqdm import tqdm
import argparse
import os
from joblib import Parallel, delayed

def convert_stl_to_obj(stl_file, obj_file):
    # Read STL file
    try:
        stl_mesh = mesh.Mesh.from_file(stl_file)
    except Exception as e:
        print(f"Error occurred while processing {stl_file}: {e}")
        return None

    # Open OBJ file for writing
    with open(obj_file, 'w') as obj:
        # Write each vertex
        for v in np.unique(stl_mesh.vectors.reshape(-1, 3), axis=0):
            obj.write(f"v {v[0]} {v[1]} {v[2]}\n")

        if stl_mesh.vectors.shape[0] > 30000:
            print(stl_mesh.vectors.shape[0])
            print(stl_file)
            return None

        # Write each face
        for _, f in tqdm(enumerate(stl_mesh.vectors), total=stl_mesh.vectors.shape[0]):
            # Find the index of each vertex in the unique list of vertices
            indices = [np.where((np.unique(stl_mesh.vectors.reshape(-1, 3), axis=0) == vertex).all(axis=1))[0][0] + 1 for vertex in f]
            obj.write(f"f {indices[0]} {indices[1]} {indices[2]}\n")
    print(f"success: {obj_file}")

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="demo.stl", required=True, help="input file path")
    parser.add_argument("--output", default="demo.obj", required=True, help="output file path")
    args = parser.parse_args()
    convert_stl_to_obj(args.input, args.output)