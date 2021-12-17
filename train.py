import torch
from NNHedge.utils import resolve_shape

def single_epoch_train(model, optimizer, trainloader, loss_func, epoch, model_type:str):
    running_loss = 0.0 
    if model_type not in ['RNN','TCN','ATTENTION','MLP']:
        raise ValueError('Please use an available type of model. Available Models: RNN | TCN | ATTENTION |MLP')

    model.train()
    for i, data in enumerate(trainloader):
        span, C, Sn, Sn_1 = data
        if model_type == 'RNN' or model_type =='TCN':
            span = resolve_shape(span)
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
