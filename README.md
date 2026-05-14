# PeSANet

PeSANet is a PyTorch implementation of the Gray-Scott 2D model code.

## Repository Layout

```text
peasnet/
  peasnet/models/          # PeSANet and Koopman neural operator modules
  peasnet/evaluation/      # Metrics
  configs/                 # Example configuration values
  examples/                # Smoke forward example
  tests/                   # Minimal import/forward sanity checks
```

## Installation

```bash
pip install -e .
```

Install PyTorch for your CUDA or CPU environment first, following the official PyTorch instructions.

## Quick Smoke Test

```bash
python examples/smoke_forward.py --grid-size 64 --steps 2 --modes 4
```

Expected output includes a sequence shape like `[1, 3, 2, 64, 64]`.

## Basic Usage

```python
import torch
from peasnet.models import PesaNet

model = PesaNet(step=10, effective_step=list(range(10)), modes_x=8, modes_y=8)
init_state = torch.randn(1, 2, 128, 128)

sequence, second_last = model(init_state)
print(sequence.shape)  # [batch, saved_steps, channels, height, width]
```

The model expects state tensors shaped `[batch, 2, height, width]`. The original experiments used 128 x 128 Gray-Scott fields with two channels.

## Citation

If you use this code, please cite the IJCAI 2025 paper:

```bibtex
@inproceedings{ijcai2025p862,
  title = {PeSANet: Physics-encoded Spectral Attention Network for Simulating PDE-Governed Complex Systems},
  author = {Wan, Han and Zhang, Rui and Wang, Qi and Liu, Yang and Sun, Hao},
  booktitle = {Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, {IJCAI-25}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor = {James Kwok},
  pages = {7751--7759},
  year = {2025},
  month = {8},
  note = {Main Track},
  doi = {10.24963/ijcai.2025/862},
  url = {https://doi.org/10.24963/ijcai.2025/862},
}
```

Paper page: https://www.ijcai.org/proceedings/2025/862

## Notes

- `peasnet.models.PesaNet` is the cleaned public version of the original `RCNN` model in `GS_KNOATTSplittest.py`.
- `peasnet.models.KNO2d` contains the spectral Koopman operator with channel attention from `KNOATTNchange.py`.
- Hard-coded checkpoint paths, CUDA device selection, plotting scripts, and baseline models were removed from the public package.
- Add a license file before publishing the repository.
