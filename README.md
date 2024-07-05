# Differential Surfel Rasterization MCMC

This project extends the [surfel-rasterization engine](https://github.com/hbb1/diff-surfel-rasterization) of 2D Gaussian Splatting, by integrating a relocation kernel based on Markov Chain Monte Carlo ([3DGS-MCMC](https://ubc-vision.github.io/3dgs-mcmc/)) principles. 

This relocation strategy enhances handling Gaussian splat parameters, focusing on maintaining sample state probabilities during heuristic moves like 'move', 'split', 'clone', 'prune', and 'add'.

***Except for the relocation kernel, the engine is exactly same as the original.***

### Installation 
- If you want to use the engine, you should reinstall the diff-surfel-rasterization within this project
```bash
git clone https://github.com/hwanhuh/diff-surfel-rasterization.git
pip install . --no-cache
```
- or re-setup the python cpp extension project
```bash
python setup.py build_ext --inplace
```

### USAGE 
```python
from diff_surfel_rasterization import compute_relocation
import torch
import math

N_MAX = 51
BINOMS = torch.zeros((N_MAX, N_MAX)).float().cuda()
for n in range(N_MAX):
    for k in range(n+1):
        BINOMS[n, k] = math.comb(n, k)

def compute_relocation_cuda(opacities, scales,  ratios):
    N = opacities.shape[0]
    opacities = opacities.contiguous()
    scales = torch.cat([scales, torch.ones(scales.shape[0], 1, device=scales.device)], dim=1)
    scales = scales.contiguous()
    ratios.clamp_(min=1, max=N_MAX)
    ratios = ratios.int().contiguous()

    new_opacities, new_scales = compute_relocation(
        opacities, scales, ratios, BINOMS, N_MAX
    )
    return new_opacities, new_scales
``` 
