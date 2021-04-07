import pytest
import torch
import numpy as np
from A5.src.submission.highway import Highway


@pytest.fixture(scope='class')
def simple_highway() -> Highway:
    e_word = 3
    highway = Highway(e_word=e_word)
    # set projection weights to one and bias to 0
    highway.projection[0].weight = torch.nn.Parameter(torch.ones((e_word, e_word)))
    highway.projection[0].bias = torch.nn.Parameter(torch.zeros(e_word))
    # set gate weight to 0.3 and bias to 0
    highway.gate[0].weight = torch.nn.Parameter(torch.ones((e_word, e_word)) * 0.3)
    highway.gate[0].bias = torch.nn.Parameter(torch.zeros(e_word))

    return highway


class TestHighway:

    def test_highway(self, simple_highway):
        h = simple_highway
        batch_size = 5
        x_conv_out = torch.rand((batch_size, h.e_word))
        proj_w = torch.ones((h.e_word, h.e_word))
        proj = torch.matmul(proj_w, x_conv_out.T).T
        proj = torch.nn.functional.relu(proj)

        gate_w = torch.ones((h.e_word, h.e_word)) * 0.3
        gate = torch.matmul(gate_w, x_conv_out.T).T
        gate = torch.nn.functional.sigmoid(gate)
        un_gate = torch.ones((batch_size, h.e_word)) - gate
        expected_highway = gate * proj + un_gate * x_conv_out
        highway = h(x_conv_out)
        assert np.isclose(expected_highway.detach().numpy(), highway.detach().numpy()).all()



