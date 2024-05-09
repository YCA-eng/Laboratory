import numpy as np
import os
import argparse

def off2obj(off_file, obj_dir): 
    off = os.path.basename(off_file)
    obj = off[:-4] if off.endswith('.off') else off
    
    f = open(off_file)
    length = 0
    while "\n" in f.readline():
        length += 1
    f.close()

    head_idx = 1
    vert = []
    faces = []
    f = open(off_file)
    out = "# " + obj + "\n"
    for j in range(length):
        line = f.readline().split()

        if j == 0:
            if line[0] == "OFF":
                head_idx = 1
            elif line[0] != "OFF" and ("OFF" in line[0]):
                head_idx = 0
        
        if j > head_idx:
            y = [float(value) for value in line]
            if len(y) == 3:
                vert.append(y)
            elif len(y) == 4:
                faces.append(y[1:])
    
    vert = np.array(vert)
    max_vert = np.max(vert, axis = 0)
    min_vert = np.min(vert, axis = 0)
    cent_vert = (max_vert + min_vert) / 2
    vert = vert - cent_vert.reshape(1, 3)
    max_abs = np.max(np.abs(vert))
    scale = 0.4 / max_abs
    vert = vert * scale

    for j in range(vert.shape[0]):
        out += "v " + str(vert[j, 0]) + " " + str(vert[j, 1]) + " " + str(vert[j, 2]) + "\n"
    
    faces = np.array(faces)
    for j in range(faces.shape[0]):
        out += "f " + str(int(faces[j, 0]+1)) + " " + str(int(faces[j, 1]+1)) + " " + str(int(faces[j, 2]+1)) + "\n"
    
    w = open(os.path.join(obj_dir, "{}.obj".format(obj)), "w")  # 修改了这里
    w.write(out)
    w.close()
    f.close()
    print("Done: " + obj)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="demo.off", required=True, help="input file path")
    parser.add_argument("--output", default=".", required=True, help="output file path")
    args = parser.parse_args()
    off2obj(args.input, args.output)