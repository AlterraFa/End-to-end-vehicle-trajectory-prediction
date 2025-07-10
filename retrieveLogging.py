# %%
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

def showLogging(evalName: str):
    eventFilePishguImproved = "./logger/PishguImproved/events.out.tfevents.1734978572.alterraserver.3974.1"
    eventFilePishgu = "./logger/Pishgu/events.out.tfevents.1734809800.alterraserver.16251.1"

    event_accPishgu = EventAccumulator(eventFilePishgu)
    event_accPishgu.Reload()
    event_accPishguImproved = EventAccumulator(eventFilePishguImproved)
    event_accPishguImproved.Reload()

    scalarsPishguImproved = event_accPishguImproved.Scalars("Validation/" + evalName)
    scalarsPishgu = event_accPishgu.Scalars("Validation/" + evalName)

    steps = [s.step + 1 for s in scalarsPishguImproved]
    valuesPishguImproved = [s.value for s in scalarsPishguImproved]
    valuesPishgu = [s.value for s  in scalarsPishgu]

    plt.figure(figsize=(16, 10))
    plt.plot(steps, valuesPishguImproved, label = "PishguImproved")
    plt.plot(steps, valuesPishgu, label = "Pishgu")
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel(evalName)
    plt.title("Evaluation on metrics: " + evalName)
    plt.grid()
    plt.show()
# %%
showLogging("FDE(m)")
showLogging("ADE(m)")
showLogging("RMSE-1s")
showLogging("RMSE-2s")
showLogging("RMSE-3s")
showLogging("RMSE-4s")
showLogging("RMSE-5s")
# %%
import os, sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'PishguImproved'))
)


import torch
from torch_geometric.data.batch import Batch as tgb
from torchinfo import summary
from PishguImproved.utils.network import *
from PishguImproved.utils.util import makeGraphData

gpu = torch.device('cuda')
cpu = torch.device("cpu")

data       = torch.load("./processedDataset/testLoader.pt", weights_only = False, map_location = cpu)
model      = NetGINConv(30, 50)
modelState = torch.load("./PishqguImproved/model/model_No.100.pt", weights_only = False, map_location = cpu)
model.load_state_dict(modelState)


observedTraj, predTrajGT, observedTrajRel, lossMask, seqStartEnd = next(iter(data))
graphData  = makeGraphData(observedTraj, observedTrajRel, seqStartEnd)
graphBatch = tgb.from_data_list(graphData)

print("MODEL SUMMARY")
summary(model, input_data = [observedTrajRel, graphBatch.x, graphBatch.edge_index])
# %%