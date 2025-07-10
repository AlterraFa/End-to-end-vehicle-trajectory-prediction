# Some of the functions in this file based on the following git repository: https://github.com/agrimgupta92/sgan
# These include relative_to_abs, l2_loss, displacement_error, and final_displacement_error

# It is for operations from the data presented in thier paper Social-GAN https://arxiv.org/abs/1803.10892

# The paper is cited as follows:
#@inproceedings{gupta2018social,
#  title={Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks},
#  author={Gupta, Agrim and Johnson, Justin and Fei-Fei, Li and Savarese, Silvio and Alahi, Alexandre},
#  booktitle={IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
#  number={CONF},
#  year={2018}
#}

#util.py
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch as tgb
from tqdm import tqdm
import gc


def rel2Abs(relTraj: torch.Tensor, startPos: torch.Tensor):
    # batch, seqLen, 2
    displacement = torch.cumsum(relTraj, dim = 1)
    return displacement + startPos

def writterInit(writter, model, dataLoader: DataLoader, device):
    observedTraj, _, observedTrajRel, _, seqStartEnd = next(iter(dataLoader))
    graphData = makeGraphData(observedTraj, observedTrajRel, seqStartEnd)
    graphBatch = tgb.from_data_list(graphData)
    writter.add_graph(model.to(device), (observedTrajRel.to(device), graphBatch.x.to(device), graphBatch.edge_index.to(device)))
    writter.flush()
    writter.close()

def l2(predTraj: torch.Tensor, predTrajGT: torch.Tensor, lossMask: torch.Tensor):
    # batch, seqLen, 2
    loss = (lossMask[:, :, None] * predTraj - predTrajGT) ** 2
    dy = loss.sum(dim = 0) / loss.shape[0]
    return dy, torch.sum(loss) / torch.prod(torch.tensor(lossMask.shape))

def adeEval(predTraj: torch.Tensor, predTrajGT: torch.Tensor, lossMask: torch.Tensor) -> float:
    score = ((((predTraj - predTrajGT) ** 2).sum(-1) ** .5 * lossMask).mean(-1)).mean()
    return score.detach().item()

def fdeEval(predTraj: torch.Tensor, predTrajGT: torch.Tensor) -> float:
    score = (((predTraj - predTrajGT) ** 2)[:, -1, :].sum(dim = -1) ** .5).mean()
    return score.detach().item()

def rmseEval(predTraj: torch.Tensor, predTrajGT: torch.Tensor, timeStep: int) -> float:
    score = (((predTraj - predTrajGT) ** 2)[:, timeStep, :].sum(dim = -1) ** .5).mean()
    return score.detach().item()

def makeGraphData(observedTraj: torch.Tensor, observedTrajRel: torch.Tensor, seqStartEnd: torch.Tensor) -> list[Data]: #type: ignore
    dataGraph = []
    
    # Iterate over the sequences
    for (start, end) in seqStartEnd:
        # Number of nodes in the current sequence
        numNodes = int((end - start).item())
        
        # Get observed and relative trajectories
        N = observedTraj[start: end, :, :, :].reshape(numNodes, -1)
        deltaN = observedTrajRel[start: end, :, :, :].reshape(numNodes, -1)
        x = torch.hstack((N, deltaN))  # Concatenate N and deltaN
        
        # Generate the edge list dynamically for the current sequence
        uniqueNodeIdx = torch.arange(0, numNodes, 1).unsqueeze(1)
        edgeList1 = uniqueNodeIdx.tile(1, numNodes - 1).reshape(-1)
        
        # Generate edgeList2 without self-loops
        edgeList2 = uniqueNodeIdx.squeeze().repeat(numNodes, 1)
        mask = torch.eye(numNodes, dtype=torch.bool)
        edgeList2[mask] = -1
        edgeList2 = edgeList2[edgeList2 != -1].reshape(-1)
        
        edgeIdx = torch.vstack([edgeList1, edgeList2]).to(torch.long)
        
        # Create Data object for this sequence
        data = Data(x = x, edge_index = edgeIdx, num_nodes = numNodes)
        dataGraph.append(data)  # Append the Data object
    
    return dataGraph


def val(model, valLoader: DataLoader, device: torch.device, epoch: int, writter = None):
    observedLen = 15 #type: ignore
    model.eval()
    losses, adeScores, fdeScores, rmseScores = [], [], [], []
    lossTotal = 0
    timeStepPart = 25 // 5
    with tqdm(valLoader, unit = "batch", desc = f"Epoch: {epoch + 1} - Evaluating") as tepoch:
        for idx, batch in enumerate(tepoch):

            observedTraj, predTrajGT, observedTrajRel, lossMask, seqStartEnd = batch
            predTrajGT = predTrajGT.squeeze()
            graphData = makeGraphData(observedTraj, observedTrajRel, seqStartEnd)
            graphBatch = tgb.from_data_list(graphData) #type: ignore

            with torch.no_grad():
                predTrajRel = model(observedTrajRel.to(device), graphBatch.x.to(device), graphBatch.edge_index.to(device)).to(torch.device('cpu')) #type: ignore
                predTrajRel = predTrajRel.reshape(predTrajRel.shape[0], -1, 2) # Relative trajectory between 2 adjacent timestep
                
                predTraj = rel2Abs(predTrajRel, observedTraj[:, :, -1, :])
    
                lossMask = lossMask[:, observedLen: ]
                _, loss = l2(predTraj, predTrajGT, lossMask)
                adeScore = adeEval(predTraj, predTrajGT, lossMask)
                fdeScore = fdeEval(predTraj, predTrajGT)
                rmseScore = [round(rmseEval(predTraj, predTrajGT, timeStepPart * i), 2) for i in range(5)]
                
                
                lossTotal += loss.item()
                tepoch.set_postfix(
                    gpuMemory = f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB",
                    valLoss = lossTotal / (idx + 1),
                    ADE = f"{adeScore:.2f}(m)",
                    FDE = f"{fdeScore:.2f}(m)",
                    RMSE = f"{rmseScore}(m)"
                )

                
                losses += [loss.item()]
                adeScores += [adeScore]
                fdeScores += [fdeScore]
                rmseScores += [rmseScore]
            gc.collect()
            del graphBatch, graphData
            
    lossAvg = torch.mean(torch.tensor(losses)).item()
    adeAvg = torch.mean(torch.tensor(adeScores)).item()
    fdeAvg = torch.mean(torch.tensor(fdeScores)).item()
    rmseAvg = torch.mean(torch.tensor(rmseScores), dim = 0)
    
    if writter is not None:
        writter.add_scalar("Validation/Loss", lossAvg, epoch)
        writter.add_scalar("Validation/ADE(m)", adeAvg, epoch)
        writter.add_scalar("Validation/FDE(m)", fdeAvg, epoch)

        for i, rmse in enumerate(rmseAvg):
            writter.add_scalar(f"Validation/RMSE-{i + 1}s", rmse, epoch)
            
        writter.flush()
    torch.cuda.empty_cache()
    return losses


def train(model, trainLoader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, epoch: int, writter = None):
    observedLen = 15 #type: ignore
    model.train()
    losses, gpuMems = [], []
    lossTotal = 0
    with tqdm(trainLoader, unit = "batch", desc = f"Epoch: {epoch + 1} - Training") as tepoch:
        for idx, batch in enumerate(tepoch):
            optimizer.zero_grad()

            # DATA PREP
            observedTraj, predTrajGT, observedTrajRel, lossMask, seqStartEnd = batch
            predTrajGT = predTrajGT.squeeze()
            graphData = makeGraphData(observedTraj, observedTrajRel, seqStartEnd)
            graphBatch = tgb.from_data_list(graphData) #type: ignore
            
            # TRAINING
            predTrajRel = model(observedTrajRel.to(device), graphBatch.x.to(device), graphBatch.edge_index.to(device)).to(torch.device('cpu')) #type: ignore # This outputs (Cars, (predTraj * 2))
            predTrajRel = predTrajRel.reshape(predTrajRel.shape[0], -1, 2) # Relative trajectory between 2 adjacent timestep
            predTraj = rel2Abs(predTrajRel, observedTraj[:, :, -1, :])
            

            lossMask = lossMask[:, observedLen: ]
            dy, loss = l2(predTraj, predTrajGT, lossMask)
            

            dy.backward(gradient=torch.ones_like(dy))
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            lossTotal += loss.item()
            tepoch.set_postfix(
                gpuMemory=f"{torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB",
                trainLoss = lossTotal / (idx + 1)
            )
            losses += [loss.item()]
            gpuMems += [torch.cuda.max_memory_allocated(device) / 1024**3]
            gc.collect()
            del graphBatch, graphData
            
    lossAvg = torch.mean(torch.tensor(losses)).item()
    gpuAvg = torch.mean(torch.tensor(gpuMems)).item()
    if writter is not None:
        writter.add_scalar("Train/Loss", lossAvg, epoch)
        writter.add_scalar("Train/GPU", gpuAvg, epoch)

        writter.flush()
            
    torch.cuda.empty_cache()
    return losses

def pred(model, observedTraj: torch.Tensor, observedTrajRel: torch.Tensor, device: torch.device):
    seqStartEnd = torch.tensor([(0, len(observedTraj))])
    graphData = makeGraphData(observedTraj, observedTrajRel, seqStartEnd)
    graphBatch = tgb.from_data_list(graphData) #type: ignore

    predTrajRel = model(observedTrajRel.to(device), graphBatch.x.to(device), graphBatch.edge_index.to(device)).to(torch.device('cpu')) #type: ignore
    predTrajRel = predTrajRel.reshape(predTrajRel.shape[0], -1, 2) # Relative trajectory between 2 adjacent timestep
    predTraj = rel2Abs(predTrajRel, observedTraj[:, :, -1, :])
        
    gc.collect()
    torch.cuda.empty_cache()
    return predTraj
