import os
import argparse

def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

def rescale_obj(obj_path, obj_scaled_path, scale):
    with open(obj_path, 'r') as source:
        with open(obj_scaled_path, 'w') as target:
            for line in source:
                target_line = line

                if(line.startswith('v ')):
                    coordinates = [float(coordinate) for coordinate in line.split(' ')[1:]]
                    rescaled = [c * scale for c in coordinates]
                    rescaled_as_str = " ".join([str(c) for c in rescaled])
                    target_line = f'v {rescaled_as_str}\n'
                target.write(target_line)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="demo.obj",required=True, help="input file path")
    parser.add_argument("--scale", default=0.2, type=float, required=True, help="scale factor")
    args = parser.parse_args()

    input_filename = os.path.basename(args.input)
    input_base, input_ext = os.path.splitext(input_filename)
    output_filename = f"{input_base}_scaled{input_ext}"
    output_path = os.path.join(os.getcwd(), output_filename)

    rescale_obj(args.input, output_path, args.scale)