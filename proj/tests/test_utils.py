import pytest
import numpy as np
import typing as t
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import keras as krs
from gaaml import utils as util
from gaaml import consts as const
from gaaml.classes.NetIndividual import NetIndividual as _NI

mark__test_cr_net_from_ind=pytest.mark.parametrize(
  ('input_output', 'gens_strs', 'exp_batch', 'exp_epoch', 'exp_optimalizer_type', 'exp_activation_types', 'exp_weight_shapes'),
  [
    ((2, 4), (
      '0000001  0101000111  0010111011  1  111100000011  1100110100  0101010110',
      '1101001001  0011101110',
      '10 01 11',
    ), 821, 343, krs.optimizers.Adam, [
      krs.activations.sigmoid, # 2
      krs.activations.tanh, # 1
      krs.activations.leaky_relu, # 3
    ], [
      (2, 587), # i->1h
      (587,), # +w^
      (587, 240), # 1h->2h
      (240,), # +w^
      (240, 4), # 2h->o
      (4,), # +w^
    ]),
    ((50, 99), (
      '0000001  0101000111  0010111011  1  111100000011  1100010100  1100110100',
      '1101001001  0011101110',
      '10 00 11',
    ), 789, 821, krs.optimizers.Adam, [
      krs.activations.sigmoid, # 2
      krs.activations.relu, # 0
      krs.activations.leaky_relu, # 3
    ], [
      (50, 587), # i->1h
      (587,), # +w^
      (587, 240), # 1h->2h
      (240,), # +w^
      (240, 99), # 2h->o
      (99,), # +w^
    ]),
    ((5, 4), (
      '0000010  0101000111  0010111011  0  111100000011  1001010011  0010011001',
      '1101001001  0011101110',
      '10 01 11',
    ), 596, 154, krs.optimizers.SGD, [
      krs.activations.sigmoid, # 2
      krs.activations.tanh, # 1
      krs.activations.leaky_relu, # 3
      krs.activations.sigmoid, # 2
    ], [
      (5, 587), # i->1h
      (587,), # +w^
      (587, 240), # 1h->2h
      (240,), # +w^
      (240, 62), # 2h->3h
      (62,), # +w^
      (62, 4), # 3h->o
      (4,), # +w^
    ]),
    ((2, 4), (
      '1111111  0101000111  0010111011  0  111100000011  1100110100  1100110100',
      '1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111',
      '00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00',
    ), 821, 821, krs.optimizers.SGD, [
      krs.activations.relu, # 0
    ]*129, [
      (2, 724), # i->h_1
      (724,), # +w_1^
      *([
        (724, 724), # h_{n-1}->h_n
        (724,), # +w_n^
      ]*127),
      (724, 4), # h_n->o
      (4,), # +w_o^
    ]),
  ]
)
@mark__test_cr_net_from_ind
def test_cr_net_from_ind(
  input_output: tuple[int, int],
  gens_strs: tuple[str, str, str],
  exp_batch: int,
  exp_epoch: int,
  exp_optimalizer_type: type[krs.optimizers.Optimizer],
  exp_activation_types: list[t.Callable],
  exp_weight_shapes: list[tuple[int, ...]],
) -> None:
  """
        layer_num  num_seed   type_seed  opt learning_rate   epochs    batch_size
  bits0: 0000001  0101000111  0010111011  1  111100000011  1100110100  0101010110
  bits_n: 1101001001  0011101110
  bits_t: 10  01  11

  expected:
  fenotype0: {
    len:   1+1=2
    n_s:   327+0=327
    t_s:   187+0=187
    opt:   1+0=1
    l_r:   3843+0=3843
    epoch: 820+1=821
    batch_size: 820+1=821
  }
  fenotype1: [841+2->843>724 -> 585+2->587, 238+2->240]
  fenotype2: [2+0->2, 1+0->1, 3+0->3]
  """
  # values
  ni=_NI(
    const.BIN_PART_LIST_LEN,
    const.BIN_PART_NEURON_NUM_SEED,
    const.BIN_PART_NEURON_TYPE_SEED,
    (
      const.BIN_PART_REST,
      const.NEURON_NUM,
      const.NEURON_TYPE,
    ),
  )
  for i, gen_str in enumerate(gens_strs):
    ni.gen[i].gen._gen=bytearray(gen_str.replace(' ', '').encode())
    ni.gen[i]._update_fenotype()
  ni._update()

  # for i, gen_str in enumerate(gens_strs):
  #   print(gen_str, len(ni.gen[i].fenotype), ni.gen[i].fenotype)

  # test
  ret=util.cr_net_from_ind(ni, *input_output)

  # results
  assert isinstance(ret, tuple)
  assert len(ret)==3
  model, batch_size, epochs=ret
  assert isinstance(model, krs.Model)
  assert isinstance(batch_size, int)
  assert isinstance(epochs, int)
  assert isinstance(model.optimizer, exp_optimalizer_type)
  weights: list[np.ndarray]=model.get_weights()
  # print([w.shape for w in weights])
  assert isinstance(weights, list)
  assert len(weights)==len(exp_weight_shapes)
  assert all(isinstance(w, np.ndarray) for w in weights)
  assert all(w.shape==ws for w, ws in zip(weights, exp_weight_shapes))
  layers: list[krs.layers.Dense]=model.layers
  # print([l.activation for l in layers])
  assert len(layers)==len(exp_activation_types)
  assert all(l.activation is a for l,a in zip(layers, exp_activation_types))
