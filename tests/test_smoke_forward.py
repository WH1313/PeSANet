import torch

from peasnet.models import KNO2d, PesaNet


def test_kno2d_forward_shape():
    model = KNO2d(op_size=8, modes_x=4, modes_y=4)
    x = torch.randn(1, 2, 16, 16)
    y = model(x)
    assert y.shape == x.shape


def test_peasnet_forward_shape():
    model = PesaNet(step=2, effective_step=[0, 1], modes_x=4, modes_y=4)
    x = torch.randn(1, 2, 64, 64)
    sequence, second_last = model(x)
    assert sequence.shape == (1, 3, 2, 64, 64)
    assert second_last is not None
    assert second_last.shape == x.shape


def test_peasnet_state_dict_keeps_original_core_key_names():
    model = PesaNet(step=1, effective_step=[0], modes_x=4, modes_y=4)
    keys = set(model.state_dict())
    assert "kno.enc.enc_conv1.weight" in keys
    assert "kno.koopman_layer.ca.ca1.fc.0.weight" in keys
    assert "kno.w0.weight" in keys
    assert "crnn_cell.Wh1_u.weight" in keys
