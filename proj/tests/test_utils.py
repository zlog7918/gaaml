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

  # test
  ret=util.cr_net_from_ind(ni, *input_output)

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
  # print([w.shape for w in weights])
  assert isinstance(weights, list)
  assert len(weights)==len(exp_weight_shapes)
  assert all(isinstance(w, np.ndarray) for w in weights)
  assert all(w.shape==ws for w, ws in zip(weights, exp_weight_shapes))
  layers: list[krs.layers.Dense]=model.layers
  assert len(layers)==len(exp_activation_types)
  assert all(l.activation is a for l,a in zip(layers, exp_activation_types))

mark__test___is_categorical=pytest.mark.parametrize(
  ('data', 'dtype', 'exp_is_categorial'),
  [
    ([['afsd'], ['fsdfd']], None, True), # one column with only strings
    # ([['afsd'], ['fsdfd']], np.object_, True), # one column with only strings
    ([['afsd'], ['fsdfd']], np.dtypes.ObjectDType, True), # one column with only strings
    ([[0], [2]], None, None), # one column with only ints
    ([[0], [2]], np.dtypes.ObjectDType, None), # one column with only ints
    ([[0.], [2.]], None, False), # one column with any other combination
    ([[0.], [2.]], np.dtypes.ObjectDType, False), # one column with any other combination
    # ([['afsd', 'jljdas'], ['fsdfd', 'lhsalhds']], None, False),
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
@mark__test___is_categorical
def test___is_categorical(data: list[list[t.Any]], dtype: type[np.dtype]|None, exp_is_categorial: bool|None):
  # values
  data_np=np.array(data, dtype=dtype)

  # test
  is_categorial=util.__is_categorical(data_np)

  # results
  assert is_categorial==exp_is_categorial

mark__test_get_fit_func_without_validation=pytest.mark.parametrize(
  ('training_data', 'test_data', 'number_of_attributes', 'gens_strs'),
  [
    (np.zeros((5, 2)), np.zeros((2, 2)), -1, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 00',
    )),
    (np.zeros((10, 3)), np.zeros((8, 3)), -1, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    (np.array([ # sin(x_1)+cos(x_2)
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
    ]), np.array([
      [4, 2, -1.1729493318550706],
      [4, 3, -1.7467949919083736],
      [4, 6, 0.20336779134243776],
      [5, 6, 0.0012460119872275133],
    ]), -1, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 00',
    )),
    (np.array([ # sin(x_1)+cos(x_2) and cos(x_1)+log(x_2)
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
    ]), np.array([
      [4, 2, -1.1729493318550706, 0.039503559696333346],
      [4, 3, -1.7467949919083736, 0.44496866780449784],
      [4, 6, 0.20336779134243776, 1.1381158483644431],
      [5, 6, 0.0012460119872275133, 2.075421654691281],
    ]), -2, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 00',
    )),
  ]
)
@mark__test_get_fit_func_without_validation
def test_get_fit_func_without_validation(
  tmp_path: Path,
  training_data: np.ndarray,
  test_data: np.ndarray,
  number_of_attributes: int,
  gens_strs: tuple[str, str, str],
) -> None:
  # values
  # TODO
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
  ret=util.get_fit_func(training_data, None, test_data, number_of_attributes)
  _ret=ret(ni, tmp_path, 0) # type: ignore

  # results
  assert isinstance(ret, t.Callable)
  assert isinstance(float(_ret), float)

  assert (tmp_path/'model_meta.data').exists()
  assert (tmp_path/'model.weights.h5').exists()
  # assert _ret==pytest.approx(expected_fit)

mark__test_get_fit_func_with_validation=pytest.mark.parametrize(
  ('training_data', 'validation_data', 'test_data', 'number_of_attributes', 'gens_strs'),
  [
    (np.zeros((5, 2)), np.zeros((3, 2)), np.zeros((2, 2)), -1, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 00',
    )),
    (np.zeros((10, 3)), np.zeros((5, 3)), np.zeros((8, 3)), -1, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    (np.array([ # sin(x_1)+cos(x_2)
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
    ]), np.array([
      [2, 2, 0.4931505902785393],
      [2, 3, -0.0806950697747637],
      [2, 6, 1.8694677134760478],
      [5, 3, -1.9489167712635838],
    ]), np.array([
      [4, 2, -1.1729493318550706],
      [4, 3, -1.7467949919083736],
      [4, 6, 0.20336779134243776],
      [5, 6, 0.0012460119872275133],
    ]), -1, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 00',
    )),
    (np.array([ # sin(x_1)+cos(x_2) and cos(x_1)+log(x_2)
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
    ]), np.array([
      [2, 2, 0.4931505902785393, 0.2770003440128029],
      [2, 3, -0.0806950697747637, 0.6824654521209674],
      [2, 6, 1.8694677134760478, 1.3756126326809126],
      [5, 3, -1.9489167712635838, 1.382274474131336],
    ]), np.array([
      [4, 2, -1.1729493318550706, 0.039503559696333346],
      [4, 3, -1.7467949919083736, 0.44496866780449784],
      [4, 6, 0.20336779134243776, 1.1381158483644431],
      [5, 6, 0.0012460119872275133, 2.075421654691281],
    ]), -2, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 00',
    )),
  ]
)
@mark__test_get_fit_func_with_validation
def test_get_fit_func_with_validation(
  tmp_path: Path,
  training_data: np.ndarray,
  validation_data: np.ndarray,
  test_data: np.ndarray,
  number_of_attributes: int,
  gens_strs: tuple[str, str, str],
) -> None:
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
  # lambda ni: ni.gen[0].fenotype['x'], ni
  ret=util.get_fit_func(training_data, validation_data, test_data, number_of_attributes)
  _ret=ret(ni, tmp_path, 0) # type: ignore

  # results
  assert isinstance(ret, t.Callable)
  assert isinstance(float(_ret), float)
  assert (tmp_path/'model_meta.data').exists()
  assert (tmp_path/'model.weights.h5').exists()

mark__test_error_get_fit_func_not_same_training_vs_test=pytest.mark.parametrize(
  ('training_data', 'validation_data', 'test_data', 'number_of_attributes', 'gens_strs'),
  [
    (np.zeros((10, 7)), None, np.zeros((8, 4)), -1, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    (np.zeros((8, 7)), None, np.zeros((8, 4)), -1, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    (np.zeros((8, 7)), np.zeros((8, 4)), np.zeros((8, 4)), -1, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    (np.zeros((8, 7)), np.zeros((8, 7)), np.zeros((8, 4)), -1, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    (np.zeros((8, 7)), None, np.zeros((8, 6)), -2, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    (np.zeros((8, 7)), np.zeros((10, 6)), np.zeros((8, 6)), -2, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    (np.zeros((8, 7)), np.zeros((10, 7)), np.zeros((8, 6)), -2, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
  ]
)
@mark__test_error_get_fit_func_not_same_training_vs_test
def test_error_get_fit_func_not_same_training_vs_test(
  training_data: np.ndarray,
  validation_data: np.ndarray|None,
  test_data: np.ndarray,
  number_of_attributes: int,
  gens_strs: tuple[str, str, str],
) -> None:
  # values
  # TODO
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
  # lambda ni: ni.gen[0].fenotype['x'], ni
  with pytest.raises(ValueError) as excinfo:
    _=util.get_fit_func(training_data, validation_data, test_data, number_of_attributes)

  # results
  assert str(excinfo.value)=='training_data and test_data do not have the same number of attributes in data or output'

mark__test_error_get_fit_func_not_same_validation=pytest.mark.parametrize(
  ('training_data', 'validation_data', 'test_data', 'number_of_attributes', 'gens_strs'),
  [
    (np.zeros((8, 7)), np.zeros((8, 6)), np.zeros((10, 7)), -1, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    (np.zeros((8, 7)), np.zeros((10, 8)), np.zeros((10, 7)), -1, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    (np.zeros((8, 6)), np.zeros((8, 8)), np.zeros((8, 6)), -2, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    (np.zeros((8, 6)), np.zeros((8, 5)), np.zeros((8, 6)), -2, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
    (np.zeros((8, 6)), np.zeros((8, 5)), np.zeros((8, 6)), -3, (
      '0000001  0101000111  0010111011  1  111100000011  0000110100  0000110100',
      '0000001001  0000000110',
      '10 01 11',
    )),
  ]
)
@mark__test_error_get_fit_func_not_same_validation
def test_error_get_fit_func_not_same_validation(training_data: np.ndarray, validation_data: np.ndarray, test_data: np.ndarray, number_of_attributes: int, gens_strs: tuple[str, str, str]) -> None:
  # values
  # TODO
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
  # lambda ni: ni.gen[0].fenotype['x'], ni
  with pytest.raises(ValueError) as excinfo:
    _=util.get_fit_func(training_data, validation_data, test_data, number_of_attributes)

  # results
  assert str(excinfo.value)=='validation_data does not have the same number of attributes in data or output as training_data and test_data'

if __name__=='__main__':
  from pathlib import Path
  import sys, inspect, tempfile
  ret_tuple=tuple[tuple[t.Any, ...], dict[str, t.Any]]
  def get_args_kwargs_from_mark(mark: pytest.MarkDecorator, index: int|slice=slice(None)) -> list[ret_tuple]:
    names_args_kwargs: tuple[str, list[t.Any]]|tuple[tuple[str, ...], list[tuple[t.Any, ...]]]=mark.args
    names, args_kwargs=names_args_kwargs
    if isinstance(index, int):
      args_kwargs=[args_kwargs[index]]
    else:
      args_kwargs=args_kwargs[index]
    return [(
      (), (
        {names: a_kw}
          if isinstance(names, str) else
        {n: v for n,v in zip(names, a_kw)}
      )
    ) for a_kw in args_kwargs]
  args_kwargs=(((),{}),)
  func_name=sys.argv[1]
  func=locals()[func_name]
  mark=f'mark__{func_name}'
  print_index=lambda i, args, kwargs: print('no index')
  if mark in locals():
    mark=locals()[mark]
    print_index=lambda i, args, kwargs: print(f'index {i} with: {(args, kwargs)}')
    try:
      index=int(sys.argv[2]),
      print_index=(lambda index: (
        lambda i, args, kwargs: print(f'index {index} with: {(args, kwargs)}')
      ))(index[0])
    except IndexError:
      index=()
    args_kwargs=get_args_kwargs_from_mark(mark, *index)
  flags={
    'monkeypatch',
    'tmp_path',
  }
  params=inspect.signature(func).parameters
  flags={k: k in params for k in flags}
  for i, (args, kwargs) in enumerate(args_kwargs):
    print_index(i, args, kwargs)
    after_tests: list[t.Callable[[], None]]=[]
    if flags['monkeypatch']:
      kwargs['monkeypatch']=pytest.MonkeyPatch()
    if flags['tmp_path']:
      tmp_path=tempfile.TemporaryDirectory()
      after_tests.append(lambda: tmp_path.cleanup())
      kwargs['tmp_path']=Path(tmp_path.name)
    # print(args, kwargs)
    try:
      func(*args, **kwargs)
    finally:
      for a in after_tests:
        a()
