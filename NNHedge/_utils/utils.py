import torch 
import numpy as np 
import matplotlib.pyplot as plt
from .._instrunments.instrunments import geometric_brownian_motion, price_call_BS
from .data import generate_span_dataset
from .._train.dataloader import SpanDataset, get_dataloader


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
        plt.savefig(f'{PATH}/{MODEL_NAME}_{epoch+1}.png', bbox_inches = 'tight')
        plt.clf()

def model_to_onnx(model, span_length=3, S0=100, r=0.02, sigma=0.2, T=1, ts=1/22, n_path=10000 ):
    X_asset = geometric_brownian_motion(S0, r, sigma, T, ts, n_path)
    X_call  = price_call_BS(X_asset, K, T/ts, r, sigma)

    span_X, span_C = generate_span_dataset(X_asset,X_call, span_length = span_length)
    ds = SpanDataset(span_X,span_C)
    trainloader = get_dataloader(ds, shuffle=True, drop_last = True)

    batch = next(iter(trainloader))[0].reshape(256,-1)
    yhat = net(batch)
    input_names = ['SPAN']
    output_names = ['Delta']
    torch.onnx.export(net, batch, 'model.onnx', input_names=input_names, output_names=output_names)

    
@torch.no_grad()
def extract_weight(model,testloader):
    total_weight = []
    for data in testloader:
        data = data[0]
        for span in data:
            X = net.query(span)
            weight = F.softmax(X).reshape(-1,1)
            weighted_layer = net.value.weight.detach() * weight
            sum = torch.sum(weighted_layer,dim=0).detach().numpy()
            total_weight.append(sum)
    total_weight = torch.tensor(total_weight)
    return F.softmax(torch.mean(total_weight,dim=0))
