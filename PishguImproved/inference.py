import torch
from utils.trajectories import *
from utils.network import *
from utils.util import *

device = torch.device('cuda')
testLoader = torch.load("../processedDataset/testLoader.pt", weights_only = False)

observed_steps = 15
prediction_steps = 25
points_per_position = 2
outputSize = points_per_position * prediction_steps
numFeatures = points_per_position * observed_steps

device = torch.device('cuda')
modelState = torch.load("./model/model_No.100.pt", weights_only = False)
model = NetGINConv(30, 50).to(device)
model.load_state_dict(modelState)

val(model, testLoader, device, 0)