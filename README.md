# mupify

Important notes:
 * In your architecture, use ReLU layers (i.e., use nn.ReLU() and not torch.functional.relu)
 * Not intended for optimizers other than SGD
 * The functions to use are mupify(model, optimizer, param) and rescale(model, gamma). See documentation in mupify.py and examples in example.ipynb.