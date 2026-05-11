# Model Notes

## Source Mapping

The public code was refactored from:

- `GS_KNOATTSplittest.py`
- `KNOATTNchange.py`

The active model path in the source script used the `RCNN` class, which combines:

- a Gray-Scott recurrent cell with RK4 integration,
- a 2D Koopman neural operator update,
- complex channel attention in the Fourier domain.

In this package, that model is exposed as `peasnet.models.PesaNet`. The alias `peasnet.models.RCNN` is kept for easier migration from the original script.

## Removed From Public Release

The following were not copied:

- generated `.mat` and `.npy` data files,
- trained `.pt` checkpoints,
- log files,
- result figures and result `.mat` files,
- baseline model files such as FNO/FFNO/LFNO/ConvLSTM variants,
- hard-coded local paths and CUDA device selection,
- one-off plotting scripts tied to local result directories.

## Compatibility Notes

The model state dict keeps the original main submodule names `kno` and `crnn_cell`, so checkpoints from the original `RCNN` model are easier to adapt. The public `forward` returns a tensor sequence by default. Use `model(init_state, return_list=True)` to recover the original list-style output.
