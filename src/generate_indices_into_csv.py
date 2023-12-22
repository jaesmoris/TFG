import numpy as np
import trimesh
import os
import random

# ID, PATH, ORIGIN, ROW, LANDRACE, FOLD, TRAIN/VALIDATION/TEST, IS_FARO, IS_UNKNOWN

# PATHS
DATASET_PATH = "/home/msiau/data/tmp/jesmoris/dummy_dataset_csv"
CSV_PATH = "/home/msiau/workspace/asdf/indices.csv"

# PARAMETERS
FOLD_SIZE = 0.2
#TRAIN_SIZE = 0.6
#VALIDATION_SIZE = 0.2
#TEST_SIZE = 0.2

random.seed(42)
CONT = 0

def grain_info(grain_path):
    is_faro = 0
    is_unknown = 0
    is_bere = 0
    origin = "Dundee" if "Dundee" in grain_path else "Orkney"
    row = 2 if "2ROW" in grain_path else 6
    if "British" in grain_path:
        landrace = "British"
    elif "Scottish" in grain_path:
        landrace = "Scottish"
    elif "Scandinavian" in grain_path:
        landrace = "Scandinavian"
    elif "BERE Orkney" in grain_path or "BERE ORKNEY" in grain_path:
        landrace = "BERE Orkney"
        is_bere = 1
    elif "BERE Unknown" in grain_path:
        landrace = "BERE Unknown"
        is_bere = 1
        is_unknown = 1
    elif "BERE Western Isles" in grain_path:
        landrace = "BERE Western Isles"
        is_bere = 1
    elif "Faro" in grain_path:
        landrace = "Faro"
        is_faro = 1
    else:
        landrace = "ERROR"
    return origin, row, landrace, is_bere, is_faro, is_unknown

def generate_csv(current_path):
    global CONT
    # List available paths
    paths_in_current_path = os.listdir(current_path)
    paths_in_current_path.sort()
    
    # Check if there are folders
    more_folders = False
    for path in paths_in_current_path:
        if os.path.isdir(current_path + "/" + path):
            more_folders = True
            generate_csv(current_path + "/" + path)

    if not more_folders:
        # Compute indices
        folder_size = len(paths_in_current_path)
        folder_indices = [i for i in range(folder_size)]
        random.shuffle(folder_indices)
        
        # Split indices
        splitter = int(FOLD_SIZE * folder_size)
        kfold_0 = folder_indices[0:splitter+1]
        kfold_1 = folder_indices[splitter+1:2*splitter+1]
        kfold_2 = folder_indices[2*splitter+1:3*splitter+1]
        kfold_3 = folder_indices[3*splitter+1:4*splitter+1]
        kfold_4 = folder_indices[4*splitter+1:]
        
        # Arrange lists for nested loops
        partition_list = ["train", "train", "train", "validation", "test"]
        partition_indices_list = [kfold_0, kfold_1, kfold_2, kfold_3, kfold_4]
        
        # Nested loops
        for partition_index, partition_indices in enumerate(partition_indices_list):
            for i in partition_indices:
                # Define csv entries
                grain_id = CONT + i
                grain_path = current_path[len(DATASET_PATH):] + "/" + paths_in_current_path[folder_indices[i]]
                origin, row, landrace, is_bere, is_faro, is_unknown = grain_info(grain_path)
                k_fold = partition_index
                partition = partition_list[partition_index]
                with open(CSV_PATH, 'a') as csv_file:
                    csv_file.write(f"{grain_id},{grain_path},{origin},{row},{landrace},{is_bere},{is_faro},{is_unknown},{k_fold},{partition}\n")
        
        CONT += folder_size


with open(CSV_PATH, 'w') as csv_file:
    csv_file.write("INDEX,GRAIN_PATH,ORIGIN,ROW,LANDRACE,IS_BERE,IS_FARO,IS_UNKNOWN,KFOLD,TRAIN(0|1|2)/VALIDATION(3)/TEST(4)\n")

generate_csv(DATASET_PATH)
