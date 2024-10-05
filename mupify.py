from torch import nn
import torch
import numpy as np
import types

def set_multiplier(layer, g):
    if not hasattr(layer, '_base_fwd'):
        layer._base_fwd = layer.forward
    layer._multiplier = g
    new_fwd = lambda slf, x: slf._multiplier * slf._base_fwd(x)
    layer.forward = types.MethodType(new_fwd, layer)

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
def get_param(layer_type, param, layer_din, width):
    assert layer_type in ["readin", "hidden", "readout"]
    param = param.lower()
    assert param in ["ntp", "mup", "mfp", "ntp-lr", "mup-lr"]

    activity = np.sqrt(width) if param in ["mup", "mfp", "mup-lr"] else 1    
    if layer_type in ["readin", "hidden"]:
        q = activity**2/layer_din
    elif layer_type == "readout":
        q = 1/layer_din
    
    if param in ['mup-lr', 'ntp-lr']:
        g = 1
        lr = q
        scale = np.sqrt(q) / activity
    elif param == 'mup':
        g = np.sqrt(q)
        lr = 1
        scale = 1 / activity
    elif param in ['mfp', 'ntp']:
        g = np.sqrt(q) / activity
        lr = activity**2
        scale = 1
    assert (activity * scale)**2 - lr < 1e-10
    assert (g*scale*activity)**2 - q  < 1e-10
    return g, lr, scale

# Tag the readin and readout layers; compute model width and tag it to the model
def mark_anatomy(model, verbose):
    if hasattr(model, '_modelwidth'):
        return
    widths = []
    layers = []
    for k, v in model.named_modules():
        if not hasattr(v, 'weight'):
            continue
        assert type(v) in [nn.Linear, nn.Conv2d], f"Can't handle module {k} of type {v}"
        chan_out, chan_in = v.weight.shape[0], v.weight.shape[1]
        if len(layers) == 0:
            widths.append(chan_in)
        widths.append(chan_out)
        layers.append((k, v))

    assert len(layers) > 1, f"Model must be deeper"    
    layers[0][1]._layertype = "readin"
    layers[-1][1]._layertype = "readout"
    for _, layer in layers[1:-1]:
        layer._layertype = "hidden"
    model._modelwidth = np.max(widths[1:-1])

    if verbose:
        print("== Model anatomy ==")
        print(f"d_in = {widths[0]}")
        print(f"d_out = {widths[-1]}")
        print(f"widths: {widths[1:-1]}")
        print(f"\t using width = {model._modelwidth}")
        print(f"readin layer: {layers[0][0]}")
        print(f"readout layer: {layers[-1][0]}")
        print()


def mupify(model, optimizer, param, verbose=False):
    """
    Reinitializes a model+optimizer in-place to a chosen parameterization. Function does not return.
    
    Params:
        model (nn.Module): model whose layer multipliers and init weights to mupify.
        optimizer (optim.SGD): SGD optimizer. (Other optimizers not tested and probably don't work.)
        param (str): One of ["ntp", "mup", "mfp", "ntp-lr", "mup-lr"].
        verbose (bool): If True, prints model anatomy.
    """
    mark_anatomy(model, verbose)
    for k, v in model.named_modules():
        if type(v) == nn.ReLU:
            set_multiplier(v, np.sqrt(2))
        if not hasattr(v, 'weight'):
            continue
        wt_d_in = np.prod(v.weight.shape[1:])
        g, lr, scale = get_param(v._layertype, param, wt_d_in, model._modelwidth)
        set_multiplier(v, g)
        set_init_scale(v, scale)
        set_lr(optimizer, v, lr)


def rescale(model, gamma):
    """
    Rescales the outputs of the model by 1/gamma, i.e, gamma>1 shrinks the outputs.
    """
    mark_anatomy(model, verbose=False)
    readout = None
    for _, v in model.named_modules():
        if hasattr(v, '_layertype') and v._layertype == "readout":
            readout = v
    assert hasattr(readout, '_multiplier')
    readout._rescale = readout._multiplier / gamma
    new_fwd = lambda slf, x: slf._rescale * slf._base_fwd(x)
    readout.forward = types.MethodType(new_fwd, readout)
        
