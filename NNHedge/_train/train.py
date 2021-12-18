import torch
from .._utils.utils import resolve_shape
import numpy as np

def single_epoch_train(model, optimizer, trainloader, loss_func, epoch, model_type:str, K=100):
    running_loss = 0.0 
    if model_type not in ['RNN','TCN','ATTENTION','SpanMLP']:
        raise ValueError('Please use an available type of model. Available Models: RNN | TCN | ATTENTION |MLP')

    model.train()
    for i, data in enumerate(trainloader):
        span, C, Sn, Sn_1 = data
        if model_type == 'RNN' or model_type =='TCN':
            span = torch.unsqueeze(span,-1)
        C    = resolve_shape(C)
        Sn   = resolve_shape(Sn)
        Sn_1 = resolve_shape(Sn_1)
        optimizer.zero_grad()

        outputs = model(span)

        loss = loss_func(outputs * (Sn - Sn_1) - C + torch.max(Sn - K, torch.zeros(256, 1)), torch.zeros(256, 1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print('[%d] loss: %.6f' % (epoch + 1, running_loss))   

@torch.no_grad()
def single_epoch_test(model, testloader, model_type:str, K=100):
    y_val = []
    for i, data in enumerate(testloader):
        span, C, Sn, Sn_1 = data
        if model_type == 'RNN' or model_type =='TCN':
            span = torch.unsqueeze(span,-1)
        C    = resolve_shape(C)
        Sn   = resolve_shape(Sn)
        Sn_1 = resolve_shape(Sn_1)
        output = model(span)
        output = output * (Sn - Sn_1) - C + torch.max(Sn - K, torch.zeros(100, 1))

        y_val.extend(output.detach().tolist())
    return np.array(y_val).reshape(-1)
