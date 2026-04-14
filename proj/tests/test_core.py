import sys
import pytest
import numpy as np
import typing as t
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import gaaml.core as core
from gaaml import consts as const
from gaaml.classes.NetIndividual import NetIndividual as _NI
from gaaml.classes.Generations import Generations as _G

class MockPlt:
  def __init__(self) -> None:
    self.figures: dict[int, dict[str, t.Any]]={}
    self.curr_fig: int|None=None
    self.figure_calls: int=0
    self.plot_calls: int=0
    self.show_calls: int=0
  def figure(self, num: int, *_: t.Any, **__: t.Any) -> None:
    self.figures[num]={}
    self.curr_fig=num
    self.figure_calls+=1
  def plot(self, *_: t.Any, **__: t.Any) -> None:
    self.plot_calls+=1
  def show(self, *_: t.Any, **__: t.Any) -> None:
    self.show_calls+=1
  def xlabel(self, *_: t.Any, **__: t.Any) -> None: ...
  def ylabel(self, *_: t.Any, **__: t.Any) -> None: ...
  def xlim(self, xlim: tuple, *_: t.Any, **__: t.Any) -> None:
    assert self.curr_fig is not None
    self.figures[self.curr_fig]['xlim']=xlim
  def ylim(self, ylim: tuple, *_: t.Any, **__: t.Any) -> None:
    assert self.curr_fig is not None
    self.figures[self.curr_fig]['ylim']=ylim
  def title(self, *_: t.Any, **__: t.Any) -> None: ...

@pytest.mark.parametrize(
  ('ni_str', 'expected_fit'),
  [
    ('0 0  1 0  1 1  1 0 1 0 1', 16.),
    ('0 0  1 0  1 1  1 1 1 1 1', 25.),
    ('0 0  1 0  1 1  1 1 1 1 0', 25.),
    ('0 0  1 0  1 1  0 0 0 1 0', 0.),
    ('0 0  1 0  1 1  0 0 0 0 0', 0.),
    ('0 0  1 0  1 1  0 0 1 0 0', 0.),
    ('0 0  1 0  1 1  0 0 1 0 1', 0.),
    ('0 0  1 0  1 1  0 0 1 1 0', 1.),
  ]
)
def test_fitness(ni_str: str, expected_fit: float) -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (5, -5, 25)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3
  ni=_NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))
  ni.gen[0].gen._gen=bytearray(ni_str.replace(' ', '').encode())
  ni.gen[0]._update_fenotype()

  # test
  ret=core.fitness(lambda ni: ni.gen[0].fenotype['x'], ni)

  # results
  assert isinstance(float(ret), float)
  assert ret==pytest.approx(expected_fit)

mark__test___f_without_validation=pytest.mark.parametrize(
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
@mark__test___f_without_validation
def test___f_without_validation(
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
  # lambda ni: ni.gen[0].fenotype['x'], ni
  ret=core.__f(training_data, None, test_data, number_of_attributes)
  with pytest.warns() as warninfo:
    _ret=ret(ni, 0) # type: ignore

  # results
  assert len(warninfo.list)==3
  assert all(isinstance(warn.message, DeprecationWarning) for warn in warninfo.list)
  assert all(
    str(warn.message).find('__array__ implementation doesn\'t accept a copy keyword, so passing copy=False failed')>=0
      for warn in
    warninfo.list
  )
  assert isinstance(ret, t.Callable)
  assert isinstance(float(_ret), float)
  print(f'calculated fittness: {_ret}')
  # assert _ret==pytest.approx(expected_fit)

mark__test___f_with_validation=pytest.mark.parametrize(
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
@mark__test___f_with_validation
def test___f_with_validation(
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
  ret=core.__f(training_data, validation_data, test_data, number_of_attributes)
  with pytest.warns() as warninfo:
    _ret=ret(ni, 0) # type: ignore

  # results
  assert len(warninfo.list)==3
  assert all(isinstance(warn.message, DeprecationWarning) for warn in warninfo.list)
  assert all(
    str(warn.message).find('__array__ implementation doesn\'t accept a copy keyword, so passing copy=False failed')>=0
      for warn in
    warninfo.list
  )
  assert isinstance(ret, t.Callable)
  assert isinstance(float(_ret), float)
  print(f'calculated fitness: {_ret}')
  # assert _ret==pytest.approx(expected_fit)

mark__test_cr_network=pytest.mark.parametrize(
  'number_of_generations',
  [1, 2, 5]
)
@mark__test_cr_network
def test_cr_network(monkeypatch: pytest.MonkeyPatch, number_of_generations: int) -> None:
  # values
  training_data=np.zeros((2, 2))
  test_data=np.zeros((2, 2))
  population_size=10
  monkeypatch.setattr(core, "__f", lambda *args, **kwargs: lambda x: 1)

  # test
  ret=core.cr_network(
    training_data,
    test_data,
    population_size=population_size,
    number_of_generations=number_of_generations,
  )

  # results
  assert isinstance(ret, _G)
  assert ret.curr_generations==number_of_generations

@pytest.mark.parametrize(
  'number_of_generations',
  [1, 2, 5]
)
def test_cr_network_plot(monkeypatch: pytest.MonkeyPatch, number_of_generations: int) -> None:
  # values
  import matplotlib
  mock_plt=MockPlt()
  monkeypatch.setattr(matplotlib, "pyplot", mock_plt)
  monkeypatch.setattr(core, "__f", lambda *args, **kwargs: lambda x: 1)

  training_data=np.zeros((2, 2))
  test_data=np.zeros((2, 2))
  population_size=10

  # test
  ret=core.cr_network(
    training_data,
    test_data,
    population_size=population_size,
    number_of_generations=number_of_generations,
    plot=True,
  )

  # results
  assert isinstance(ret, _G)
  assert ret.curr_generations==number_of_generations
  assert mock_plt.figure_calls==3
  assert len(mock_plt.figures)==3
  assert mock_plt.plot_calls==3
  assert mock_plt.show_calls==3
  for k in mock_plt.figures.keys():
    assert mock_plt.figures[k]['xlim']==(0, number_of_generations+2)
  for k in mock_plt.figures.keys():
    assert mock_plt.figures[k]['ylim']==(0, 1)

@pytest.mark.parametrize(
  'number_of_generations',
  [1, 2, 5]
)
def test_cr_network_plot_0_in_fitnesses(monkeypatch: pytest.MonkeyPatch, number_of_generations: int) -> None:
  # values
  import matplotlib
  mock_plt=MockPlt()
  monkeypatch.setattr(matplotlib, "pyplot", mock_plt)
  monkeypatch.setattr(core, "__f", lambda *args, **kwargs: lambda x: 0)

  training_data=np.zeros((2, 2))
  test_data=np.zeros((2, 2))
  population_size=10

  # test
  ret=core.cr_network(
    training_data,
    test_data,
    population_size=population_size,
    number_of_generations=number_of_generations,
    plot=True,
  )

  # results
  assert isinstance(ret, _G)
  assert ret.curr_generations==number_of_generations
  assert mock_plt.figure_calls==3
  assert len(mock_plt.figures)==3
  assert mock_plt.plot_calls==3
  assert mock_plt.show_calls==3
  for k in mock_plt.figures.keys():
    assert mock_plt.figures[k]['xlim']==(0, number_of_generations+2)
  for k in mock_plt.figures.keys():
    assert mock_plt.figures[k]['ylim']==(0, .5)

mark__test_error___f_not_same_training_vs_test=pytest.mark.parametrize(
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
@mark__test_error___f_not_same_training_vs_test
def test_error___f_not_same_training_vs_test(
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
    _=core.__f(training_data, validation_data, test_data, number_of_attributes)

  # results
  assert str(excinfo.value)=='training_data and test_data do not have the same number of attributes in data or output'

mark__test_error___f_not_same_validation=pytest.mark.parametrize(
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
@mark__test_error___f_not_same_validation
def test_error___f_not_same_validation(training_data: np.ndarray, validation_data: np.ndarray, test_data: np.ndarray, number_of_attributes: int, gens_strs: tuple[str, str, str]) -> None:
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
    _=core.__f(training_data, validation_data, test_data, number_of_attributes)

  # results
  assert str(excinfo.value)=='validation_data does not have the same number of attributes in data or output as training_data and test_data'

mark__test_error_cr_network_0generations=pytest.mark.parametrize(
  'number_of_generations',
  [0, -1, -2]
)
@mark__test_error_cr_network_0generations
def test_error_cr_network_0generations(monkeypatch: pytest.MonkeyPatch, number_of_generations: int) -> None:
  # values
  training_data=np.zeros((2, 2))
  test_data=np.zeros((2, 2))
  population_size=10
  monkeypatch.setattr(
    core,
    "__f",
    lambda *args, **kwargs: lambda x: 1,
  ) # first population also has calculated fittness (before generations obj is created)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=core.cr_network(
      training_data,
      test_data,
      population_size=population_size,
      number_of_generations=number_of_generations,
    )

  # results
  assert str(excinfo.value)=='max_num_gen: is too small, it should at least equal 1'
