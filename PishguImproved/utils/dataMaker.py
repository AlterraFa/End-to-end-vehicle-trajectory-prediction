import os, sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

import glob
import torch
import argparse
import yaml
from utils.loader import data_loader
from utils.trajectories import *
from torch.utils.data import DataLoader 


def config_reader(type: str):
    with open(f"./configs/{type}/birds_eye.yaml", "r") as f:
        data = yaml.safe_load(f)
    return data


def main(): 
    parser = argparse.ArgumentParser(
        description="Preprocess trajectory datasets and save DataSet/DataLoader tensors."
    )
    parser.add_argument(
        "-n", "--name",
        help="Name of the dataset folder",
        default="ngsim"
    )
    parser.add_argument(
        "-b", "--batch-size",
        help="Batch size for DataLoader",
        type=int,
        default=64
    )
    parser.add_argument(
        "-w", "--workers",
        help="Number of workers for DataLoader",
        type=int,
        default=10
    )
    args = parser.parse_args()
    
    if args.name == "ngsim":
        data = config_reader("vehicles")
    else:
        data = config_reader("pedestrian")
    
    observed_steps = data['input_data']['observed_steps']
    prediction_steps = data['input_data']['prediction_step']
    points_per_position = data['input_data']['points_per_position']
    outputSize = points_per_position * prediction_steps
    numFeatures = points_per_position * observed_steps
    

    base_path = os.path.join("../datasets", args.name)
    if not os.path.isdir(base_path):
        print(f"ERROR: Can't find `{base_path}`")
        sys.exit(1)
    
    for datasetPath in glob.glob(os.path.join(base_path, "*")):
        datasetNameType = os.path.basename(datasetPath)
        
        print(f"Creating {datasetNameType} loader")
        dset, _ = data_loader(
            datasetPath, 
            batch_size = args.batch_size, 
            obs_len = observed_steps, 
            pred_len = prediction_steps, 
            delim = 'tab'
        )

        print(f"Saving complete: {datasetNameType}Dset.pt")
        torch.save(dset, f"processedDataset/{dset}Dset.pt")

        loader = DataLoader(
            dset,
            batch_size = args.batch_size,
            shuffle = True,
            num_workers = args.workers,
            collate_fn = seq_collate
        )
        torch.save(loader, f"processedDataset/{datasetNameType}Loader.pt")
        print(f"Saving complete: {datasetNameType}Loader.pt")
        
if __name__ == "__main__":
    main()