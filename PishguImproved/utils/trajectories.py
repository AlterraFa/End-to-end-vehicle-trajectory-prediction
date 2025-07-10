

# This file is based on the following git repository: https://github.com/agrimgupta92/sgan

# It loads the dataset presented in thier paper Social-GAN https://arxiv.org/abs/1803.10892

# The paper is cited as follows:
#@inproceedings{gupta2018social,
#  title={Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks},
#  author={Gupta, Agrim and Johnson, Justin and Fei-Fei, Li and Savarese, Silvio and Alahi, Alexandre},
#  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
#  number={CONF},
#  year={2018}
#}

import logging
import os
import math

import numpy as np

import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
from tqdm import tqdm

logger = logging.getLogger(__name__)


def seq_collate(data):
    (obs_seq_list, pred_seq_list, obs_seq_rel_list, loss_mask_list, _) = zip(*data)

    _len = [len(seq) for seq in obs_seq_list]
    cum_start_idx = [0] + np.cumsum(_len).tolist()
    seq_start_end = [[start, end]
                     for start, end in zip(cum_start_idx, cum_start_idx[1:])]

    obs_traj = torch.cat(obs_seq_list, dim=0).unsqueeze(1).permute(0, 1, 3, 2)
    pred_traj = torch.cat(pred_seq_list, dim=0).unsqueeze(1).permute(0, 1, 3, 2)
    obs_traj_rel = torch.cat(obs_seq_rel_list, dim=0).unsqueeze(1).permute(0, 1, 3, 2)
    loss_mask = torch.cat(loss_mask_list, dim=0)
    seq_start_end = torch.LongTensor(seq_start_end)
    out = [
        obs_traj, pred_traj, obs_traj_rel,
        loss_mask, seq_start_end
    ]

    return tuple(out)


def read_file(_path, delim='\t'):
    data = []
    if delim == 'tab':
        delim = '\t'
    elif delim == 'space':
        delim = ' '
    with open(_path, 'r') as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def poly_fit(traj, traj_len, threshold) -> bool:
    """
    Input:
    - traj: Numpy array of shape (2, traj_len)
    - traj_len: Len of trajectory
    - threshold: Minimum error to be considered for non linear traj
    Output:
    - int: 1 -> Non Linear 0-> Linear
    """
    t = np.linspace(0, traj_len - 1, traj_len)
    # It returns residual?
    res_x = np.polyfit(t, traj[0, -traj_len:], 2, full=True)[1]
    res_y = np.polyfit(t, traj[1, -traj_len:], 2, full=True)[1]
    if res_x + res_y >= threshold:
        return 1.0
    else:
        return 0.0


class TrajectoryDataset(Dataset):
    """Dataloder for the Trajectory datasets"""
    def __init__(
        self, data_dir, obs_len=8, pred_len=12, skip=1, threshold=0.002,
        min_ped=1, delim='\t'
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDataset, self).__init__()


        # The data is sorted from vehicle ID first then frame ID
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim

        all_files = os.listdir(self.data_dir)
        all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        seq_list_rel = []
        loss_mask_list = []
        seq_list_id = []
        for idx, path in enumerate(all_files):
            data = read_file(path, delim)
            frames = np.unique(data[:, 0]).tolist() # Frame ID (First column)
            frame_data = []
            for frame in frames:
                frame_data.append(data[frame == data[:, 0], :]) # Find all vehicles appeared in that frame

            # Accounting for the data outside of bounds
            num_sequences = int(
                math.ceil((len(frames) - self.seq_len + 1) / skip))

            for idx in tqdm(range(0, num_sequences * self.skip + 1, skip), desc = f"File No. {idx + 1}"):
                # concatenate from current index to index + (sequence length) on row (sequence length, each vehicle information length)
                curr_seq_data = np.concatenate(
                    frame_data[idx:idx + self.seq_len], axis=0)
               
                # Find out what vehicle was presented in that frame sequence 
                peds_in_curr_seq = np.unique(curr_seq_data[:, 1])
                
                # Empty relative position of all vehicle (delta N) (Number of vehicles, 2, sequence length)
                curr_seq_rel = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                # This is absolute position with the same shape as above
                curr_seq = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                # Frame ID for what?
                curr_frame_id = np.zeros((len(peds_in_curr_seq), 2, self.seq_len))
                # Current Loss mask?
                curr_loss_mask = np.zeros((len(peds_in_curr_seq),
                                           self.seq_len))
                curr_ped_id = np.zeros((len(peds_in_curr_seq), 
                                        self.seq_len))

                # ped is pedestrian or vehicle
                num_peds_considered = 0
                for ped_id in peds_in_curr_seq:
                    # Get the current data of that vehicle in the current sequence (Number of occurence in a sequence, a vehicle info length)
                    curr_ped_seq = curr_seq_data[curr_seq_data[:, 1] ==
                                                 ped_id, :]
                    
                    # Rounding the entire array to 4 decimals
                    curr_ped_seq = np.around(curr_ped_seq, decimals=4)
                   
                    # SAFE GUARD (If there's not enough sequence of a vehicle then ignore it)
                    # curr_ped_seq[0, 0] first frame the vehicle appeared in the data
                    # After that get the index from frames
                    pad_front = frames.index(curr_ped_seq[0, 0]) - idx
                    pad_end = frames.index(curr_ped_seq[-1, 0]) - idx + 1

                    if pad_end - pad_front != self.seq_len:
                        continue
                    # SAFE GUARD
                    
                    # Transpose it so it is [[frame id], [vehicle id]]
                    curr_ped_frame_id = np.transpose(curr_ped_seq[:, :2])
                    # Permutate it so it is [[x], [y]]
                    curr_ped_seq = np.transpose(curr_ped_seq[:, 2:])
                    curr_ped_seq = curr_ped_seq
                    # Make coordinates relative
                    rel_curr_ped_seq = np.zeros(curr_ped_seq.shape)
                    rel_curr_ped_seq[:, 1:] = \
                        curr_ped_seq[:, 1:] - curr_ped_seq[:, :-1]
                    _idx = num_peds_considered
                    if curr_ped_seq.shape[1] != pad_end-pad_front: # to check if a sequence has a lost frame
                        continue
                    # Adding sequence of one vehicle to the array for all vehicle
                    curr_seq[_idx, :, pad_front:pad_end] = curr_ped_seq
                    curr_seq_rel[_idx, :, pad_front:pad_end] = rel_curr_ped_seq
                    curr_frame_id[_idx, :, pad_front:pad_end] = curr_ped_frame_id
                    
                    curr_loss_mask[_idx, pad_front:pad_end] = 1
                    curr_ped_id[_idx, pad_front: pad_end] = ped_id
                    num_peds_considered += 1
                    # if a vehicle doesn't satisfies, that node is zero

                if num_peds_considered > min_ped:
                    num_peds_in_seq.append(num_peds_considered)
                    loss_mask_list.append(curr_loss_mask[:num_peds_considered])
                    seq_list.append(curr_seq[:num_peds_considered])
                    seq_list_rel.append(curr_seq_rel[:num_peds_considered])
                    seq_list_id.append(curr_ped_id[:num_peds_considered])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        seq_list_rel = np.concatenate(seq_list_rel, axis=0)
        loss_mask_list = np.concatenate(loss_mask_list, axis=0)
        seq_list_id = np.concatenate(seq_list_id, axis = 0)

        # Convert numpy -> Torch Tensor
        self.obs_traj = torch.from_numpy(
            seq_list[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj = torch.from_numpy(
            seq_list[:, :, self.obs_len:]).type(torch.float)
        self.obs_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, :self.obs_len]).type(torch.float)
        self.pred_traj_rel = torch.from_numpy(
            seq_list_rel[:, :, self.obs_len]).type(torch.float)
        self.loss_mask = torch.from_numpy(loss_mask_list).type(torch.float)
        self.id = torch.from_numpy(seq_list_id).type(torch.int)
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :], self.pred_traj[start:end, :],
            self.obs_traj_rel[start:end, :], self.loss_mask[start:end, :], self.id[start:end, :]
        ]
        return out