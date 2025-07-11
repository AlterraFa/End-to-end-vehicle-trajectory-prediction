# %%
import torch
from utils.trajectories import *
from utils.util import *
import utils.network as net 
from torch.utils.tensorboard import SummaryWriter
from rich.traceback import install
install(show_locals = False)

device = torch.device('cuda')
trainLoader = torch.load("./processedDataset/trainLoader.pt", weights_only = False)
valLoader = torch.load("./processedDataset/valLoader.pt", weights_only = False)

observed_steps = 15
prediction_steps = 25
points_per_position = 2
outputSize = points_per_position * prediction_steps
numFeatures = points_per_position * observed_steps
writter = SummaryWriter(log_dir = "./logger/PishguImproved/")

device = torch.device('cuda')
model = net.NetGINConv(numFeatures, outputSize).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01, weight_decay = 0.0005)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = [10,30,45], gamma = 0.1)

writterInit(writter, model, trainLoader, device)

valHists = []
trainHists = []

for epoch in range(100):
    trainHistory = train(model, trainLoader, optimizer, device, epoch, writter)
    validationHistory = val(model, valLoader, device, epoch, writter)
    torch.save(model.state_dict(), f"./model/model_No.{epoch + 1}.pt")
    scheduler.step()
    print("")

writter.close()
# %%
