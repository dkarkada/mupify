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


def mupify(model, optimizer, readin, readout, width, param='mup', gamma=None):
    """
    Reinitializes a model/optimizer in-place to a chosen parameterization. Function does not return.
    
    Params:
        model (nn.Module): model whose layer multipliers and init weights to mupify.
        optimizer (optim.SGD): SGD optimizer. (Other optimizers not tested and probably don't work.)
        readin (str or nn.Module): The name of (or a pointer to) the readin layer
        readout (str or nn.Module): The name of (or a pointer to) the readout layer
        width (int): Model width. All hidden layers should have this width. For CNNs, width=channel_dim
        param (str): One of ["stp", "mup", "ntp"]. Selects one of (behaviorally-equivalent) parameterization schemes.
        gamma (float or None): Sets the activity (richness/laziness) of the net. None defaults to active training. gamma=1 is lazy.
    """
    
    if gamma is None:
        gamma = np.sqrt(width)
    for k, v in model.named_modules():
        if not hasattr(v, 'weight'):
            continue
        assert type(v) in [nn.Linear, nn.Conv2d], f"Can't handle module {k} of type {v}"
        
        if type(v) == nn.Linear:
            chan_out, chan_in = v.weight.shape
            wt_d_in = chan_in
        elif type(v) == nn.Conv2d:
            chan_out, chan_in, kh, kw = v.weight.shape
            wt_d_in = chan_in * kh * kw
            
        layer_type = "hidden"
        if k==readin or v==readin:
            layer_type = "readin"
        if k==readout or v==readout:
            layer_type = "readout"
        if chan_out != width and layer_type != "readout":
            print(f"Warning: d_out of {k} should be {width}, but is {chan_out}.")
        if chan_in != width and layer_type != "readin":
            print(f"Warning: d_in of {k} should be {width}, but is {chan_in}.")

        g, lr, scale = get_param(layer_type, param, wt_d_in, gamma)
        set_multiplier(v, g)
        set_init_scale(v, scale*np.sqrt(2))
        set_lr(optimizer, v, lr)
