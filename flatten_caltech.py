import os
import shutil

caltech_path = "./caltech_dataset"  # https://www.kaggle.com/datasets/athota1/caltech101
caltech_flattened_path = "./caltech_flattened"

if not os.path.exists(caltech_flattened_path):
    os.makedirs(caltech_flattened_path)

sub_dirs = [x[0] for x in os.walk(caltech_path)][1:]

for subdir in sub_dirs:
    for file in os.listdir(subdir):
        source_file = os.path.join(subdir, file)
        destination_file = os.path.join(caltech_flattened_path, os.path.split(subdir)[-1] + "_" + file)
        shutil.copy(source_file, destination_file)
