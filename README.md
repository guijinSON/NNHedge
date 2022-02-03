# NNHedge
__NNHedge__ is a PyTorch based framework for Neural Derivative Hedging.  
The following repository was implemented to ease the experiments of our paper :  

- [Neural Networks for Derivative Hedging](https://arxiv.org/abs/2112.10084).

## Installation Guide
To build and develop from source, clone this repository via
```python
git clone https://github.com/guijinSON/NNHedge.git
```

## Quick Start
To follow along the code snippets below, we recommend that you refer to the [Colab notebook](https://colab.research.google.com/drive/1V_amf3vilYtUh7TeiJPAZiibtehNNHOq?usp=sharing).

### 1. Importing NNHedge
```python
from NNHedge.NNHedge._instrunments.instrunments import geometric_brownian_motion, price_call_BS, delta_call_BS
from NNHedge.NNHedge._utils.data import generate_span_dataset
from NNHedge.NNHedge._utils.utils import resolve_shape, plot_PnL, count_parameters, model_to_onnx
from NNHedge.NNHedge._utils.metric import evaluate_model

from NNHedge.NNHedge._train.train import single_epoch_train, single_epoch_test
from NNHedge.NNHedge._train.dataloader import SpanDataset, get_dataloader
from NNHedge.NNHedge._models.models import get_model
```
### 2. Generating Simulated Asset Movements
```python
S0 = 100
K = 100
r = 0.02
sigma = 0.2
T = 1
ts = 66
dt = T/ts 
n_path = 100
num_epochs = 5

X_asset = geometric_brownian_motion(S0, r, sigma, T, ts, n_path)
X_call  = price_call_BS(X_asset, K, T/ts, r, sigma)
```

### 3. Reshaping Underlying Assets to Spans & Loading Dataloaders
```python
span_X, span_C = generate_span_dataset(X_asset,X_call, span_length = span_length)
ds = SpanDataset(span_X,span_C)
trainloader = get_dataloader(ds, shuffle=True, drop_last = True)

X_asset_test = geometric_brownian_motion(S0, r, sigma, T, ts, 1000)
X_call_test  = price_call_BS(X_asset_test, K, T/ts, r, sigma)

span_X_test, span_C_test = generate_span_dataset(X_asset_test, X_call_test, span_length = span_length)
ds_test = SpanDataset(span_X_test, span_C_test)
testloader = get_dataloader(ds_test, batch_size = 100, shuffle=True, drop_last = True)
```

### 4. Model Specifics
NNHedge support a total of 4 Neural Models.\[SpanMLP, RNN, TCN and Attention]
```python
MODEL_TYPE='SpanMLP' # Choose from ['SpanMLP','RNN','TCN','ATTENTION']
span_length = 5 # int value smaller than ts
```

### 5. Training the Model
```python
net = get_model(MODEL_TYPE, span_length)
optimizer = optim.Adam(net.parameters(),lr=0.001)
loss_func = nn.L1Loss()

for epoch in range(num_epochs):
    single_epoch_train(net, optimizer, trainloader, loss_func, epoch, MODEL_TYPE)
```

## Citation 
You can cite our work by:
```bibtex
@misc{son2021neural,
      title={Neural Networks for Delta Hedging}, 
      author={Guijin Son and Joocheol Kim},
      year={2021},
      eprint={2112.10084},
      archivePrefix={arXiv},
      primaryClass={q-fin.CP}
}
```
