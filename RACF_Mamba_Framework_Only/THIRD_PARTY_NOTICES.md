# Upstream attribution

The supplied core code references PyTorch, torchvision's EfficientNetV2-S,
and Mamba. This architecture-only package does not redistribute those libraries,
their source implementations, binaries, pretrained weights, or datasets.
No dependency versions or model presets are selected by this package.

Upstream projects referenced by the supplied implementation:

- PyTorch: https://github.com/pytorch/pytorch
- torchvision: https://github.com/pytorch/vision
- Mamba: https://github.com/state-spaces/mamba

The Mamba and EfficientNetV2 names refer to their respective upstream methods,
not methods newly introduced by this architecture reference. Any future
concrete implementation must retain applicable upstream notices and comply
with the terms of the dependencies it actually uses.

The interfaces in `framework.py` use only Python standard-library imports.
This notice does not assign a license to the project itself.
