from torch import nn
import torch
import numpy as np

def set_multiplier(layer, g):
    if not hasattr(layer, '_base_fwd'):
        layer._base_fwd = layer.forward
    layer.forward = lambda x: g*layer._base_fwd(x)

def set_init_scale(layer, scale):
    assert layer.weight is not None
    with torch.no_grad():
        layer.weight.normal_(0, scale)
        if layer.bias is not None:
            layer.bias.normal_(0, scale)

def set_lr(optimizer, layer, lr):
    # One-time setup
    if not hasattr(optimizer, '_base_lr'):
        assert len(optimizer.param_groups) == 1
        # Save base LR and save other optimizer settings
        hypers = {}
        for k, v in optimizer.param_groups[0].items():
            if k != 'params':
                if k == 'lr':
                    optimizer._base_lr = optimizer.param_groups[0]['lr']
                hypers[k] = v
        assert hasattr(optimizer, '_base_lr')
        # Create separate param group for each param and transfer other optimizer settings
        new_pgs = []
        for param in optimizer.param_groups[0]['params']:
            pg = hypers.copy()
            pg['params'] = [param]
            new_pgs.append(pg)
        optimizer.param_groups = new_pgs
    
    # Update lr for all parameters in given layer
    lr = optimizer._base_lr * lr
    for pg in optimizer.param_groups:
        assert len(pg['params']) == 1
        if id(pg['params'][0]) in [id(p) for p in layer.parameters()]:
            pg['lr'] = lr

# Given layer and parameterization, get layer multiplier, layer LR, and layer weight scale
def get_param(layer_type, param, layer_din, gamma):
    assert layer_type in ["readin", "hidden", "readout"]
    assert param.lower() in ["stp", "mup", "ntp"]
    
    if layer_type in ["readin", "hidden"]:
        q = gamma**2/layer_din
    elif layer_type == "readout":
        q = 1/layer_din
    
    if param.lower() == 'stp':
        g = 1
        lr = q
        scale = np.sqrt(q / gamma**2)
    elif param.lower() =='mup':
        g = np.sqrt(q)
        lr = 1
        scale = 1 / gamma
    elif param.lower() == 'ntp':
        g = np.sqrt(q / gamma**2)
        lr = gamma**2
        scale = 1
    assert np.allclose((gamma * scale)**2, lr)
    assert np.allclose((g*scale*gamma)**2, q)
    return g, lr, scale


def mupify(model, optimizer, readin, readout, gamma, param='mup'):
    """
    Reinitializes a model/optimizer in-place to a chosen parameterization. Function does not return.
    
    Params:
        model (nn.Module): model whose layer multipliers and init weights to mupify.
        optimizer (optim.SGD): SGD optimizer. (Other optimizers not tested and probably don't work.)
        readin (str or nn.Module): The name of (or a pointer to) the readin layer
        readout (str or nn.Module): The name of (or a pointer to) the readout layer
        gamma (float): Sets the laziness/activity of the net. gamma=1 is lazy, gamma=sqrt(width) is active.
        param (str): One of ["stp", "mup", "ntp"]. Selects one of (behaviorally-equivalent) parameterization schemes.
    """
    for k, v in model.named_modules():
        if type(v) == nn.ReLU:
            set_multiplier(v, np.sqrt(2))
        if not hasattr(v, 'weight'):
            continue
        assert type(v) in [nn.Linear, nn.Conv2d], f"Can't handle module {k} of type {v}"
        
        if type(v) == nn.Linear:
            _, chan_in = v.weight.shape
            wt_d_in = chan_in
        elif type(v) == nn.Conv2d:
            _, chan_in, kh, kw = v.weight.shape
            wt_d_in = chan_in * kh * kw
            
        layer_type = "hidden"
        if k==readin or v==readin:
            layer_type = "readin"
        if k==readout or v==readout:
            layer_type = "readout"

        g, lr, scale = get_param(layer_type, param, wt_d_in, gamma)
        set_multiplier(v, g)
        set_init_scale(v, scale)
        set_lr(optimizer, v, lr)
