# mupify

$\mu\mathrm{P}$-ify your pytorch models! It's super easy: just initialize your model and SGD optimizer as you normally would, then pass them both to `mupify`. They'll be modified in-place so that the forward/backward passes reflect the $\mu\mathrm{P}$ training regime (or lazy regime if you choose)... modulo finite-width corrections. See `example.ipynb` for a tutorial.

Not intended for use with any of the following:
* Adaptive optimizers. (SGD + momentum and/or weight decay are fine.)
* Linear layers other than dense linear layers or 2d convolutions.
* Attention blocks

Important notes:
 * `nn.ReLU()` layers are mupified to evaluate $\mathrm{max}(0, x\sqrt{2})$ rather than $\mathrm{max}(0, x)$. To avoid this behavior, use `torch.functional.relu`
 * The user-facing functions are `mupify(model, optimizer, param)` and `rescale(model, gamma)`. See documentation in `mupify.py`.