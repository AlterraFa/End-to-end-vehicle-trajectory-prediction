import os, sys
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), 'PishguImproved'))
)

from PishguImproved.utils.trajectories import *
from PishguImproved.utils.network import *
from PishguImproved.utils.util import pred, adeEval
from matplotlib.patches import Rectangle

import torch
import matplotlib.pyplot as plt
import numpy as np
np.printoptions(suppress = True)


device     = torch.device("cuda")
data       = torch.load("./processedDataset/testDset.pt", weights_only = False)
modelState = torch.load("./PishguImproved/model/model_No.100.pt", weights_only = False, map_location = device)
model      = NetGINConv(30, 50).to(device)
model.load_state_dict(modelState)

maxValX = 0
maxValY = 0
for i in range(data.num_seq):
    obsTraj, predTraj, _, _, _ = data[i]
    tempMaxX = obsTraj[:, 1, :].max()
    tempMaxY = obsTraj[:, 0, :].max()
    if tempMaxX > maxValX: maxValX = tempMaxX
    if tempMaxY > maxValY: maxValY = tempMaxY


plt.ion()
fig, ax = plt.subplots()
width = 5
height = 1
threshHold = 350

ax.plot([], [], color='red', alpha=0.7, label="Prediction")
ax.plot([], [], color='green', alpha=0.7, label="Ground Truth")
ax.scatter([], [], c='blue', label="Observed Points")
for t in range(6500, data.num_seq):
    ax.clear()
    obsTraj, predTrajGT, obsTrajRel, lossMask, velID = data[t]

    with torch.no_grad():
        predTraj = pred(model, obsTraj.unsqueeze(1).transpose(-1, -2), obsTrajRel.unsqueeze(1).transpose(-1, -2), device = device).transpose(-1, -2)
    
    yCurr = obsTraj[:, 0, -1]
    xCurr = obsTraj[:, 1, -1]
    
    mask = xCurr < threshHold     

    for xi, yi, filter in zip(xCurr, yCurr, mask):
        if filter == True:
            rect = Rectangle((xi - width / 2, yi - height / 2), width, height,
                            color='blue', alpha=0.6)
            ax.add_patch(rect)
    
    
    gt_x = predTrajGT[:, 1, :]
    gt_y = predTrajGT[:, 0, :]

    for idx, (px, py) in enumerate(zip(gt_x, gt_y)):
        if idx == 0:
            ax.plot(px, py, color='green', alpha=0.7, label = "Ground Truth")
        else:
            ax.plot(px, py, color='green', alpha=0.7)
    
    pred_x = predTraj[:, 1, :]
    pred_y = predTraj[:, 0, :]

    for idx, (px, py) in enumerate(zip(pred_x, pred_y)):
        if idx == 0:
            ax.plot(px, py, color='red', alpha=0.7, label = "Prediction")
        else:
            ax.plot(px, py, color='red', alpha=0.7)
        
    print(f"{adeEval(predTraj.transpose(-1, -2), predTrajGT.transpose(-1, -2), lossMask[:, 15:]):.2f}", flush = True, end = '\r')
    
    
    # Plot the points
    ax.scatter(xCurr, yCurr, c='blue', label=f'Time step {t + 1}')
    ax.set_xlim(50, 175)
    ax.set_ylim(0, 30)
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_title("Trajectory Evolution")
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal")

    plt.draw()
    plt.pause(0.001)

plt.ioff()  # Turn off interactive mode
plt.show()  # Keep the final plot open

# %%
