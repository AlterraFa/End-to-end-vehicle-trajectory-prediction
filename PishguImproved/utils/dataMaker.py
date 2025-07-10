import os, sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
)

import glob
import torch
from utils.loader import data_loader
from utils.trajectories import *
from torch.utils.data import DataLoader 


observed_steps = 15
prediction_steps = 25
points_per_position = 2
outputSize = points_per_position * prediction_steps
numFeatures = points_per_position * observed_steps

path = "datasets/ngsim"
for datasetPath in glob.glob(path + "/*"):
    datasetNameType = datasetPath.split("/")[-1]
    print(f"Creating {datasetNameType} loader")
    dset, _ = data_loader(datasetPath, batch_size = 64, obs_len = 15, pred_len = 25, delim = 'tab')

    print(f"Saving complete: {datasetNameType}Dset.pt")
    torch.save(dset, f"processedDataset/{dset}Dset.pt")

    loader = DataLoader(
        dset,
        batch_size = 180,
        shuffle = True,
        num_workers = 10,
        collate_fn = seq_collate
    )
    torch.save(loader, f"processedDataset/{datasetNameType}Loader.pt")
    print(f"Saving complete: {datasetNameType}Loader.pt")