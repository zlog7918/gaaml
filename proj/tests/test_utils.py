import pytest
import numpy as np
import typing as t
import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import keras as krs
from pathlib import Path
from gaaml import utils as util
from gaaml import consts as const
from gaaml.classes.NetIndividual import NetIndividual as _NI

mark__test_cr_net_from_ind=pytest.mark.parametrize(
  ('input_output', 'gens_strs', 'categorial', 'exp_epoch', 'exp_batch', 'exp_optimalizer_type', 'exp_activation_types', 'exp_weight_shapes'),
  [
    ((2, 4), (
      '00001  0101000111  0010111011  1  111100000011  1001101  0101010110',
      '1101001001  0011101110',
      '10 01 11',
    ), False, 78, 343, krs.optimizers.Adam, [
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
    ((2, 4), (
      '00001  0101000111  0010111011  1  111100000011  0011010  1100110100',
      '1101001001  0011101110',
      '10 01 11',
    ), True, 27, 821, krs.optimizers.Adam, [
      krs.activations.sigmoid, # 2
      krs.activations.tanh, # 1
      krs.activations.softmax, # last is replaced with softmax
    ], [
      (2, 587), # i->1h
      (587,), # +w^
      (587, 240), # 1h->2h
      (240,), # +w^
      (240, 4), # 2h->o
      (4,), # +w^
    ]),
    ((50, 99), (
      '00001  0101000111  0010111011  1  111100000011  1000101  1100010100',
      '1101001001  0011101110',
      '10 00 11',
    ), False, 70, 789, krs.optimizers.Adam, [
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
      '00010  0101000111  0010111011  0  111100000011  0101001  0010011001',
      '1101001001  0011101110',
      '10 01 11',
    ), False, 42, 154, krs.optimizers.SGD, [
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
    ((5, 4), (
      '00010  0101000111  0010111011  0  111100000011  1010011  0010011001',
      '1101001001  0011101110',
      '10 01 11',
    ), True, 84, 154, krs.optimizers.SGD, [
      krs.activations.sigmoid, # 2
      krs.activations.tanh, # 1
      krs.activations.leaky_relu, # 3
      krs.activations.softmax, # last is replaced with softmax
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
      '11111  0101000111  0010111011  0  111100000011  0110100  1100110100',
      '1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111  1111111111',
      '00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00  00',
    ), False, 53, 821, krs.optimizers.SGD, [
      krs.activations.relu, # 0
    ]*33, [
      (2, 724), # i->h_1
      (724,), # +w_1^
      *([
        (724, 724), # h_{n-1}->h_n
        (724,), # +w_n^
      ]*31),
      (724, 4), # h_n->o
      (4,), # +w_o^
    ]),
  ]
)
@mark__test_cr_net_from_ind
def test_cr_net_from_ind(
  input_output: tuple[int, int],
  gens_strs: tuple[str, str, str],
  categorial: bool,
  exp_batch: int,
  exp_epoch: int,
  exp_optimalizer_type: type[krs.optimizers.Optimizer],
  exp_activation_types: list[t.Callable],
  exp_weight_shapes: list[tuple[int, ...]],
) -> None:
  """
      layer_num  num_seed   type_seed  opt learning_rate  epochs  batch_size
  bits0: 00001  0101000111  0010111011  1  111100000011  1001101  0101010110
  bits_n: 1101001001  0011101110
  bits_t: 10  01  11

  expected:
  fenotype0: {
    len:   1+1=2
    n_s:   327+0=327
    t_s:   187+0=187
    opt:   1+0=1
    l_r:   3843+0=3843
    epoch: 77+1=78
    batch_size: 342+1=342
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

  # test
  ret=util.cr_net_from_ind(ni, *input_output, categorial)

  # results
  assert isinstance(ret, tuple)
  assert len(ret)==3
  model, batch_size, epochs=ret
  assert isinstance(model, krs.Model)
  assert isinstance(batch_size, int)
  assert isinstance(epochs, int)
  assert batch_size==exp_batch
  assert epochs==exp_epoch
  assert isinstance(model.optimizer, exp_optimalizer_type)
  weights: list[np.ndarray]=model.get_weights()
  assert isinstance(weights, list)
  assert len(weights)==len(exp_weight_shapes)
  assert all(isinstance(w, np.ndarray) for w in weights)
  assert all(w.shape==ws for w, ws in zip(weights, exp_weight_shapes))
  layers: list[krs.layers.Dense]=model.layers
  assert len(layers)==len(exp_activation_types)
  assert all(l.activation is a for l,a in zip(layers, exp_activation_types))

mark__test___detect_type_group=pytest.mark.parametrize(
  ('data', 'dtype', 'exp_type'),
  [
    ([['afsd'], ['fsdfd']], None, 'str'), # one column with only strings
    # ([['afsd'], ['fsdfd']], np.object_, 'str'), # one column with only strings
    ([['afsd'], ['fsdfd']], np.dtypes.ObjectDType, 'str'), # one column with only strings
    ([[b'afsd'], [b'fsdfd']], None, None), # one column with only strings
    ([[b'afsd'], [b'fsdfd']], np.dtypes.ObjectDType, None), # one column with only strings
    ([[0.], [2]], None, None), # one column with only ints
    ([[0], [2.]], np.dtypes.ObjectDType, None), # one column with only ints
    ([[0], [2]], None, 'int'), # one column with only ints
    ([[0], [2]], np.dtypes.ObjectDType, 'int'), # one column with only ints
    ([[0.], [2.]], None, None), # one column with any other combination
    ([[0.], [2.]], np.dtypes.ObjectDType, None), # one column with any other combination
    ([[0, 1], [1, 0]], None, 'int01'), # multicolumn with only 0 or 1 (exactly one 1 in row)
    ([[0, 1], [1, 0]], np.dtypes.ObjectDType, 'int01'), # multicolumn with only 0 or 1 (exactly one 1 in row)
    ([[0, 1, 0], [1, 0, 0]], None, 'int01'), # multicolumn with only 0 or 1 (exactly one 1 in row)
    ([[0, 1, 0], [1, 0, 0]], np.dtypes.ObjectDType, 'int01'), # multicolumn with only 0 or 1 (exactly one 1 in row)
    ([[False, True], [True, False]], None, 'bool'), # multicolumn with only True or False (exactly one True in row)
    ([[False, True], [True, False]], np.dtypes.ObjectDType, 'bool'), # multicolumn with only True or False (exactly one True in row)
    ([[False, False, True], [False, True, False]], None, 'bool'), # multicolumn with only True or False (exactly one True in row)
    ([[False, False, True], [False, True, False]], np.dtypes.ObjectDType, 'bool'), # multicolumn with only True or False (exactly one True in row)
    ([[1, 1], [1, 0]], None, 'int01'), # multicolumn with any other combination
    ([[1, 1], [1, 0]], np.dtypes.ObjectDType, 'int01'), # multicolumn with any other combination
    ([[0, 5], [2, 4]], None, 'int'), # multicolumn with any other combination
    ([[0, 5], [2, 4]], np.dtypes.ObjectDType, 'int'), # multicolumn with any other combination
    ([[0., 5.], [2., 4.]], None, None), # multicolumn with any other combination
    ([[0., 5.], [2., 4.]], np.dtypes.ObjectDType, None), # multicolumn with any other combination
    ([[True, True], [True, False]], None, 'bool'), # multicolumn with any other combination
    ([[True, True], [True, False]], np.dtypes.ObjectDType, 'bool'), # multicolumn with any other combination
    ([['afsd', 'gsgrs'], ['fsdfd', 'fsdfdbdgffs']], None, 'str'), # one column with only strings
    # ([['afsd', 'gsgrs'], ['fsdfd', 'fsdfdbdgffs']], np.object_, 'str'), # one column with only strings
    ([['afsd', 'gsgrs'], ['fsdfd', 'fsdfdbdgffs']], np.dtypes.ObjectDType, 'str'), # one column with only strings
  ]
)
@mark__test___detect_type_group
def test___detect_type_group(data: list[list[t.Any]], dtype: type[np.dtype]|None, exp_type: str|None):
  # values
  data_np=np.array(data, dtype=dtype)

  # test
  detected_type=util.__detect_type_group(data_np)

  # results
  assert detected_type==exp_type

mark__test___is_categorial=pytest.mark.parametrize(
  ('data', 'dtype', 'exp_is_categorial'),
  [
    ([['afsd'], ['fsdfd']], None, False), # one column with only strings
    # ([['afsd'], ['fsdfd']], np.object_, False), # one column with only strings
    ([['afsd'], ['fsdfd']], np.dtypes.ObjectDType, False), # one column with only strings
    ([[0], [2]], None, False), # one column with only ints
    ([[0], [2]], np.dtypes.ObjectDType, False), # one column with only ints
    ([[0.], [2.]], None, False), # one column with any other combination
    ([[0.], [2.]], np.dtypes.ObjectDType, False), # one column with any other combination
    ([[0, 1], [1, 0]], None, None), # multicolumn with only 0 or 1 (exactly one 1 in row)
    ([[0, 1], [1, 0]], np.dtypes.ObjectDType, None), # multicolumn with only 0 or 1 (exactly one 1 in row)
    ([[0, 1, 0], [1, 0, 0]], None, None), # multicolumn with only 0 or 1 (exactly one 1 in row)
    ([[0, 1, 0], [1, 0, 0]], np.dtypes.ObjectDType, None), # multicolumn with only 0 or 1 (exactly one 1 in row)
    ([[False, True], [True, False]], None, True), # multicolumn with only True or False (exactly one True in row)
    ([[False, True], [True, False]], np.dtypes.ObjectDType, True), # multicolumn with only True or False (exactly one True in row)
    ([[False, False, True], [False, True, False]], None, True), # multicolumn with only True or False (exactly one True in row)
    ([[False, False, True], [False, True, False]], np.dtypes.ObjectDType, True), # multicolumn with only True or False (exactly one True in row)
    ([[1, 1], [1, 0]], None, False), # multicolumn with any other combination
    ([[1, 1], [1, 0]], np.dtypes.ObjectDType, False), # multicolumn with any other combination
    ([[0, 5], [2, 4]], None, False), # multicolumn with any other combination
    ([[0, 5], [2, 4]], np.dtypes.ObjectDType, False), # multicolumn with any other combination
    ([[0., 5.], [2., 4.]], None, False), # multicolumn with any other combination
    ([[0., 5.], [2., 4.]], np.dtypes.ObjectDType, False), # multicolumn with any other combination
    ([[True, True], [True, False]], None, False), # multicolumn with any other combination
    ([[True, True], [True, False]], np.dtypes.ObjectDType, False), # multicolumn with any other combination
  ]
)
@mark__test___is_categorial
def test___is_categorial(data: list[list[t.Any]], dtype: type[np.dtype]|None, exp_is_categorial: bool|None):
  # values
  data_np=np.array(data, dtype=dtype)

  # test
  is_categorial=util.__is_categorial(data_np)

  # results
  assert is_categorial==exp_is_categorial

mark__test_get_fit_func=pytest.mark.parametrize(
  ('training_data', 'validation_data', 'test_data', 'number_of_attributes', 'is_categorial', 'gens_strs'),
  [
    (np.zeros((10, 3)).tolist(), None, np.zeros((8, 3)).tolist(), -1, False, (
      '00001  0101000111  0010111011  1  111100000011  0110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    ([ # sin(x_1)+cos(x_2)
      [5, 4, -1.6125678955267504],
      [4, 5, -0.47314030984470196],
      [1, 5, 1.1251331702711227],
      [1, 4, 0.18782736394428456],
      [1, 3, -0.1485215117925489],
      [1, 2, 0.4253241482607541],
      [1, 1, 1.3817732906760363],
      [5, 1, -0.4186219687949987],
      [4, 1, -0.21650018943978844],
      [3, 1, 0.681422313928007],
      [2, 1, 1.4495997326938215],
      [2, 4, 0.25565380596206977],
      [3, 4, -0.5125236128037447],
      [6, 4, -0.9330591190625378],
      [6, 5, 0.004246687264300386],
      [5, 5, -0.6752620891999122],
    ], None, [
      [4, 2, -1.1729493318550706],
      [4, 3, -1.7467949919083736],
      [4, 6, 0.20336779134243776],
      [5, 6, 0.0012460119872275133],
    ], -1, False, (
      '00001  0101000111  0010111011  1  111100000011  0110100  0000110100',
      '0000001001  0000000110',
      '10 01 00',
    )),
    ([ # sin(x_1)+cos(x_2) and cos(x_1)+log(x_2)
      [5, 4, -1.6125678955267504, 1.6699565465831168],
      [4, 5, -0.47314030984470196, 0.9557942915704883],
      [1, 5, 1.1251331702711227, 2.14974021830224],
      [1, 4, 0.18782736394428456, 1.9265966669880303],
      [1, 3, -0.1485215117925489, 1.6389145945362495],
      [1, 2, 0.4253241482607541, 1.2334494864280852],
      [1, 1, 1.3817732906760363, 0.5403023058681398],
      [5, 1, -0.4186219687949987, 0.28366218546322625],
      [4, 1, -0.21650018943978844, -0.6536436208636119],
      [3, 1, 0.681422313928007, -0.9899924966004454],
      [2, 1, 1.4495997326938215, -0.4161468365471424],
      [2, 4, 0.25565380596206977, 0.9701475245727482],
      [3, 4, -0.5125236128037447, 0.39630186451944516],
      [6, 4, -0.9330591190625378, 2.3464646477702567],
      [6, 5, 0.004246687264300386, 2.5696081990844664],
      [5, 5, -0.6752620891999122, 1.8931000978973265],
      [5, 4, -1.6125678955267504, 1.6699565465831168],
    ], None, [
      [4, 2, -1.1729493318550706, 0.039503559696333346],
      [4, 3, -1.7467949919083736, 0.44496866780449784],
      [4, 6, 0.20336779134243776, 1.1381158483644431],
      [5, 6, 0.0012460119872275133, 2.075421654691281],
    ], -2, False, (
      '00001  0101000111  0010111011  1  111100000011  0110100  0000110100',
      '0000001001  0000000110',
      '10 01 00',
    )),
    (np.zeros((10, 3)).tolist(), np.zeros((5, 3)).tolist(), np.zeros((8, 3)).tolist(), -1, False, (
      '00001  0101000111  0010111011  1  111100000011  0110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    ([ # sin(x_1)+cos(x_2)
      [5, 4, -1.6125678955267504],
      [4, 5, -0.47314030984470196],
      [1, 5, 1.1251331702711227],
      [1, 4, 0.18782736394428456],
      [1, 3, -0.1485215117925489],
      [1, 2, 0.4253241482607541],
      [1, 1, 1.3817732906760363],
      [5, 1, -0.4186219687949987],
      [4, 1, -0.21650018943978844],
      [3, 1, 0.681422313928007],
      [2, 1, 1.4495997326938215],
      [2, 4, 0.25565380596206977],
      [3, 4, -0.5125236128037447],
      [6, 4, -0.9330591190625378],
      [6, 5, 0.004246687264300386],
      [5, 5, -0.6752620891999122],
    ], [
      [2, 2, 0.4931505902785393],
      [2, 3, -0.0806950697747637],
      [2, 6, 1.8694677134760478],
      [5, 3, -1.9489167712635838],
    ], [
      [4, 2, -1.1729493318550706],
      [4, 3, -1.7467949919083736],
      [4, 6, 0.20336779134243776],
      [5, 6, 0.0012460119872275133],
    ], -1, False, (
      '00001  0101000111  0010111011  1  111100000011  0110100  0000110100',
      '0000001001  0000000110',
      '10 01 00',
    )),
    ([ # sin(x_1)+cos(x_2) and cos(x_1)+log(x_2)
      [5, 4, -1.6125678955267504, 1.6699565465831168],
      [4, 5, -0.47314030984470196, 0.9557942915704883],
      [1, 5, 1.1251331702711227, 2.14974021830224],
      [1, 4, 0.18782736394428456, 1.9265966669880303],
      [1, 3, -0.1485215117925489, 1.6389145945362495],
      [1, 2, 0.4253241482607541, 1.2334494864280852],
      [1, 1, 1.3817732906760363, 0.5403023058681398],
      [5, 1, -0.4186219687949987, 0.28366218546322625],
      [4, 1, -0.21650018943978844, -0.6536436208636119],
      [3, 1, 0.681422313928007, -0.9899924966004454],
      [2, 1, 1.4495997326938215, -0.4161468365471424],
      [2, 4, 0.25565380596206977, 0.9701475245727482],
      [3, 4, -0.5125236128037447, 0.39630186451944516],
      [6, 4, -0.9330591190625378, 2.3464646477702567],
      [6, 5, 0.004246687264300386, 2.5696081990844664],
      [5, 5, -0.6752620891999122, 1.8931000978973265],
      [5, 4, -1.6125678955267504, 1.6699565465831168],
    ], [
      [2, 2, 0.4931505902785393, 0.2770003440128029],
      [2, 3, -0.0806950697747637, 0.6824654521209674],
      [2, 6, 1.8694677134760478, 1.3756126326809126],
      [5, 3, -1.9489167712635838, 1.382274474131336],
    ], [
      [4, 2, -1.1729493318550706, 0.039503559696333346],
      [4, 3, -1.7467949919083736, 0.44496866780449784],
      [4, 6, 0.20336779134243776, 1.1381158483644431],
      [5, 6, 0.0012460119872275133, 2.075421654691281],
    ], -2, False, (
      '00001  0101000111  0010111011  1  111100000011  0110100  0000110100',
      '0000001001  0000000110',
      '10 01 00',
    )),
    ([
      [5, 4, 1, 0],
      [4, 5, 0, 1],
      [1, 5, 0, 1],
      [1, 4, 0, 1],
      [1, 3, 0, 1],
      [1, 2, 0, 1],
      [1, 1, 1, 0],
      [5, 1, 1, 0],
      [4, 1, 1, 0],
      [3, 1, 1, 0],
      [2, 1, 1, 0],
      [2, 4, 0, 1],
      [3, 4, 0, 1],
      [6, 4, 1, 0],
      [6, 5, 1, 0],
      [5, 5, 1, 0],
      [5, 4, 1, 0],
      [3, 3, 1, 0],
      [4, 4, 1, 0],
    ], [
      [2, 2, 1, 0],
      [2, 3, 0, 1],
      [2, 6, 0, 1],
      [5, 3, 1, 0],
    ], [
      [4, 2, 1, 0],
      [4, 3, 1, 0],
      [4, 6, 0, 1],
      [5, 6, 0, 1],
    ], -2, True, (
      '00001  0101000111  0010111011  1  111100000011  0110100  0000110100',
      '0000001001  0000000110',
      '10 01 00',
    )),
    ([
      [5, 4, 1, 0, 0],
      [4, 5, 0, 0, 1],
      [1, 5, 0, 0, 1],
      [1, 4, 0, 0, 1],
      [1, 3, 0, 0, 1],
      [1, 2, 0, 0, 1],
      [1, 1, 0, 1, 0],
      [5, 1, 1, 0, 0],
      [4, 1, 1, 0, 0],
      [3, 1, 1, 0, 0],
      [2, 1, 1, 0, 0],
      [2, 4, 0, 0, 1],
      [3, 4, 0, 0, 1],
      [6, 4, 1, 0, 0],
      [6, 5, 1, 0, 0],
      [5, 5, 0, 1, 0],
      [5, 4, 1, 0, 0],
      [3, 3, 0, 1, 0],
      [4, 4, 0, 1, 0],
    ], [
      [2, 2, 0, 1, 0],
      [2, 3, 0, 0, 1],
      [2, 6, 0, 0, 1],
      [5, 3, 1, 0, 0],
    ], [
      [4, 2, 1, 0, 0],
      [4, 3, 1, 0, 0],
      [4, 6, 0, 0, 1],
      [5, 6, 0, 0, 1],
    ], -3, True, (
      '00001  0101000111  0010111011  1  111100000011  0110100  0000110100',
      '0000001001  0000000110',
      '10 01 00',
    )),
  ]
)
@mark__test_get_fit_func
def test_get_fit_func(
  tmp_path: Path,
  training_data: list[list[object]],
  validation_data: list[list[object]]|None,
  test_data: list[list[object]],
  number_of_attributes: int,
  is_categorial: bool,
  gens_strs: tuple[str, str, str],
) -> None:
  _training_data=np.array(training_data, dtype=np.dtypes.ObjectDType)
  _validation_data=None
  if validation_data is not None:
    _validation_data=np.array(validation_data, dtype=np.dtypes.ObjectDType)
  _test_data=np.array(test_data, dtype=np.dtypes.ObjectDType)

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

  # test
  ret=util.get_fit_func(_training_data, _validation_data, _test_data, number_of_attributes, is_categorial)
  _ret=ret(ni, tmp_path)

  # results
  assert isinstance(ret, t.Callable)
  assert isinstance(float(_ret), float)
  # assert _ret==pytest.approx(expected_fit)
  assert (tmp_path/'model_meta.data').exists()
  assert (tmp_path/'model.weights.h5').exists()

mark__test_error_get_fit_func_not_2D_arr=pytest.mark.parametrize(
  ('training_data', 'validation_data', 'test_data', 'number_of_attributes'),
  [
    (np.zeros((8,)), np.zeros((8, 7)), np.zeros((8, 7)), -1),
    (np.zeros((8, 7)), np.zeros((8,)), np.zeros((8, 7)), -1),
    (np.zeros((8, 7)), np.zeros((8, 7)), np.zeros((8,)), -1),
    (np.zeros((8, 7, 3)), np.zeros((8, 7)), np.zeros((8, 7)), -1),
    (np.zeros((8, 7)), np.zeros((8, 7, 3)), np.zeros((8, 7)), -1),
    (np.zeros((8, 7)), np.zeros((8, 7)), np.zeros((8, 7, 3)), -1),
  ]
)
@mark__test_error_get_fit_func_not_2D_arr
def test_error_get_fit_func_not_2D_arr(
  training_data: np.ndarray,
  validation_data: np.ndarray|None,
  test_data: np.ndarray,
  number_of_attributes: int,
) -> None:
  # values ^

  # test
  with pytest.raises(ValueError) as excinfo:
    _=util.get_fit_func(training_data, validation_data, test_data, number_of_attributes)

  # results
  assert str(excinfo.value)=='Every data must be 2D array'

mark__test_error_get_fit_func_not_same_training_vs_test=pytest.mark.parametrize(
  ('training_data', 'validation_data', 'test_data', 'number_of_attributes'),
  [
    (np.zeros((10, 7)), None, np.zeros((8, 4)), -1),
    (np.zeros((8, 7)), None, np.zeros((8, 4)), -1),
    (np.zeros((8, 7)), np.zeros((8, 4)), np.zeros((8, 4)), -1),
    (np.zeros((8, 7)), np.zeros((8, 7)), np.zeros((8, 4)), -1),
    (np.zeros((8, 7)), None, np.zeros((8, 6)), -2),
    (np.zeros((8, 7)), np.zeros((10, 6)), np.zeros((8, 6)), -2),
    (np.zeros((8, 7)), np.zeros((10, 7)), np.zeros((8, 6)), -2),
  ]
)
@mark__test_error_get_fit_func_not_same_training_vs_test
def test_error_get_fit_func_not_same_training_vs_test(
  training_data: np.ndarray,
  validation_data: np.ndarray|None,
  test_data: np.ndarray,
  number_of_attributes: int,
) -> None:
  # values ^

  # test
  with pytest.raises(ValueError) as excinfo:
    _=util.get_fit_func(training_data, validation_data, test_data, number_of_attributes)

  # results
  assert str(excinfo.value)=='training_data and test_data do not have the same number of attributes in data or output'

mark__test_error_get_fit_func_not_same_validation=pytest.mark.parametrize(
  ('training_data', 'validation_data', 'test_data', 'number_of_attributes'),
  [
    (np.zeros((8, 7)), np.zeros((8, 6)), np.zeros((10, 7)), -1),
    (np.zeros((8, 7)), np.zeros((10, 8)), np.zeros((10, 7)), -1),
    (np.zeros((8, 6)), np.zeros((8, 8)), np.zeros((8, 6)), -2),
    (np.zeros((8, 6)), np.zeros((8, 5)), np.zeros((8, 6)), -2),
    (np.zeros((8, 6)), np.zeros((8, 5)), np.zeros((8, 6)), -3),
  ]
)
@mark__test_error_get_fit_func_not_same_validation
def test_error_get_fit_func_not_same_validation(
  training_data: np.ndarray,
  validation_data: np.ndarray,
  test_data: np.ndarray,
  number_of_attributes: int,
) -> None:
  # values ^

  # test
  with pytest.raises(ValueError) as excinfo:
    _=util.get_fit_func(training_data, validation_data, test_data, number_of_attributes)

  # results
  assert str(excinfo.value)=='validation_data does not have the same number of attributes in data or output as training_data and test_data'

mark__test_error_get_fit_func_not_valid_type=pytest.mark.parametrize(
  ('training_data', 'validation_data', 'test_data', 'number_of_attributes', 'is_categorial'),
  [
    ([
      [5.2, 4, 1, 0],
      [4, 5., 0, 1],
      [1, 5, 0, 1],
    ], [
      [2, 2.1, 0, 1],
      [2, 6, 0, 1],
      [4., 3, 1, 0],
    ], [
      [4, 2, 1, 0],
      [2, 2, 1, 1], # wrong expected value: (1, 1) can not mean categorial
      [4, 6.5, 0, 1],
    ], -2, True),
    ([
      [5.2, 4, 1, 0],
      [4, 5., 0, 1],
      [1, 5, 0, 1],
    ], None, [
      [4, 2, 1, 0],
      [2, 2, 1, 1], # wrong expected value: (1, 1) can not mean categorial
      [4, 6.5, 0, 1],
    ], -2, True),
    ([
      [5.2, 4, True, False],
      [4, 5., False, True],
      [1, 5, False, True],
    ], [
      [2, 2.1, False, True],
      [2, 6, False, True],
      [4., 3, True, False],
    ], [
      [4, 2, True, False],
      [2, 2, True, True], # wrong expected value: (True, True) can not mean categorial
      [4, 6.5, False, True],
    ], -2, True),
    ([
      [5.2, 4, True, False],
      [4, 5., False, True],
      [1, 5, False, True],
    ], None, [
      [4, 2, True, False],
      [2, 2, True, True], # wrong expected value: (True, True) can not mean categorial
      [4, 6.5, False, True],
    ], -2, True),
    ([
      [5.2, 4, True, 0],
      [4, 5., False, True],
      [1, 5, False, True],
    ], [
      [2, 2.1, False, True],
      [2, 6, False, True],
      [4., 3, True, False],
    ], [
      [4, 2, True, False],
      [2, 2, 2, False], # defaulted to first being greater
      [4, 6.5, False, True],
    ], -2, True),
    ([
      [5.2, 4, True, False],
      [4, 5., False, True],
      [1, 5, False, True],
    ], None, [
      [4, 2, True, False],
      [2, 2, 2, False], # defaulted to first being greater
      [4, 6.5, False, True],
    ], -2, True),
  ]
)
@mark__test_error_get_fit_func_not_valid_type
def test_error_get_fit_func_not_valid_type(
  training_data: list[list[object]],
  validation_data: list[list[object]]|None,
  test_data: list[list[object]],
  number_of_attributes: int,
  is_categorial: bool,
) -> None:
  # values
  _training_data=np.array(training_data, dtype=np.dtypes.ObjectDType)
  _validation_data=None
  if validation_data is not None:
    _validation_data=np.array(validation_data, dtype=np.dtypes.ObjectDType)
  _test_data=np.array(test_data, dtype=np.dtypes.ObjectDType)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=util.get_fit_func(_training_data, _validation_data, _test_data, number_of_attributes, is_categorial)

  # results
  # print(excinfo.value)
  assert str(excinfo.value)=='Given type is not valid for given data'



mark__test_error_get_fit_func_conv_not_correct=pytest.mark.parametrize(
  ('training_data', 'validation_data', 'test_data', 'number_of_attributes', 'is_categorial'),
  [
    # ([
    #   [5.2, 4, 1.1, 0], # can not be categorial 1. is float
    #   [4, 5., 0, 1],
    #   [1, 5, 0, 1],
    # ], [
    #   [2, 2.1, 0, 1],
    #   [2, 6, 0, 1],
    #   [4., 3, 1, 0],
    # ], [
    #   [4, 2, 1, 0],
    #   [2, 2, 1, 0], # defaulted to first being greater
    #   [4, 6.5, 0, 1],
    # ], -2, True),
    # ([
    #   [5.2, 4, 1.1, 0], # can not be categorial 1. is float
    #   [4, 5., 0, 1],
    #   [1, 5, 0, 1],
    # ], None, [
    #   [4, 2, 1, 0],
    #   [2, 2, 1, 0], # defaulted to first being greater
    #   [4, 6.5, 0, 1],
    # ], -2, True),
    ([
      [5.2, 4, '0'],
      [4, 5., '1'],
      [1, 5, '1'],
    ], [
      [2, 2.1, '1'],
      [2, 6, '1'],
      [4., 3, '0'],
    ], [
      [4, 2, '0'],
      [2, 2, .5], # wrong expected value: .5 is not string
      [4, 6.5, '1'],
    ], -1, True),
    ([
      [5.2, 4, '0'],
      [4, 5., '1'],
      [1, 5, '1'],
    ], None, [
      [4, 2, '0'],
      [2, 2, .5], # wrong expected value: .5 is not string
      [4, 6.5, '1'],
    ], -1, True),
    ([
      [5.2, 4, '0'],
      [4, 5., '1'],
      [1, 5, '1'],
    ], [
      [2, 2.1, '1'],
      [2, 6, '1'],
      [4., 3, '0'],
    ], [
      [4, 2, '0'],
      [2, 2, '.5'],
      [4, 6.5, '1'],
    ], -1, True), # was accepted previously, however has been simplified
    ([
      [5.2, 4, '0'],
      [4, 5., '1'],
      [1, 5, '1'],
    ], None, [
      [4, 2, '0'],
      [2, 2, '.5'],
      [4, 6.5, '1'],
    ], -1, True), # was accepted previously, however has been simplified
  ]
)
@mark__test_error_get_fit_func_conv_not_correct
def test_error_get_fit_func_conv_not_correct(
  training_data: list[list[object]],
  validation_data: list[list[object]]|None,
  test_data: list[list[object]],
  number_of_attributes: int,
  is_categorial: bool,
) -> None:
  # values
  _training_data=np.array(training_data, dtype=np.dtypes.ObjectDType)
  _validation_data=None
  if validation_data is not None:
    _validation_data=np.array(validation_data, dtype=np.dtypes.ObjectDType)
  _test_data=np.array(test_data, dtype=np.dtypes.ObjectDType)

  # test/results
  with pytest.raises(AssertionError):
    _=util.get_fit_func(_training_data, _validation_data, _test_data, number_of_attributes, is_categorial)
