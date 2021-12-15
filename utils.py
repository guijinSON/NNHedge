import torch 
import numpy as np 
import matplotlib.pyplot as plt

@torch.no_grad()
def plot_net_delta(model,train=False,epoch=None,PATH=None,MODEL_NAME=None):
    model.eval()
    input =  np.arange(0.0,1.0,0.005).reshape(-1,1)
    output = model(torch.from_numpy(input).float()).detach().numpy() 
    bs_delta = np.array([delta_call_BS(s,K,T,r,sigma) for s in np.arange(90,110,0.1)])
    if train:
        plt.plot(input, output, color = (227/255, 27/255, 35/255), label="Neural Net Delta")
        plt.plot(input, bs_delta, color = (0.0, 45/255, 106/255), label = "Black Scholes Delta")

        plt.title(f"Comparison of Deltas at Epoch {epoch+1}")
        plt.legend()
        plt.xlabel("Normalized S0")
        plt.ylabel("Call Options Delta")
        plt.savefig(f'{PATH}/{MODEL_NAME}_{epoch+1}.png')
        plt.clf()
    else:
        plt.plot(input, output, color = (227/255, 27/255, 35/255), label="Neural Net Delta")
        plt.plot(input, bs_delta, color = (0.0, 45/255, 106/255), label = "Black Scholes Delta")
        plt.xlabel("Normalized S0")
        plt.ylabel("Call Options Delta")
        plt.legend()
    model.train()
    
def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def resolve_shape(vector):
    if len(vector.shape) == 1:
        return torch.unsqueeze(vector,-1)
    return vector

def plot_PnL(X, Y, epoch, PATH=None, MODEL_NAME=None):
    plt.plot(X, Y, marker=".", linestyle='none', color = (0.0, 45/255, 106/255))
    plt.title(f"Calculated PnL at Epoch {epoch+1}")
    plt.xlabel("S0")
    plt.ylabel("Calculated PnL")
    plt.ylim([-3, 3])
    if PATH:
        plt.savefig(f'{PATH}/{MODEL_NAME}_{epoch+1}.png', bbox_inches=extent)
        plt.clf()
