import pytest
import numpy as np
import typing as t
from pathlib import Path
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import gaaml.core as core
import gaaml.utils as util
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

mark__test_fitness=pytest.mark.parametrize(
  ('ni_str', 'expected_fit'),
  [
    ('0 0  1 0  1 1  1 0 1 0 1', 1/22),
    ('0 0  1 0  1 1  1 1 1 1 1', 1/32),
    ('0 0  1 0  1 1  1 1 1 1 0', 1/31),
    ('0 0  1 0  1 1  0 0 0 1 0', 1/3),
    ('0 0  1 0  1 1  0 0 0 0 0', 1.),
    ('0 0  1 0  1 1  0 0 1 0 0', .2),
    ('0 0  1 0  1 1  0 0 1 0 1', 1/6),
    ('0 0  1 0  1 1  0 0 1 1 0', 1/7),
  ]
)
@mark__test_fitness
def test_fitness(ni_str: str, expected_fit: float) -> None:
  # values
  layers_len=('len', (2, 1, 4))
  num_seed=('num_seed', (2, 0, 3))
  type_seed=('type_seed', (2, 0, 3))
  g_schema=(
    ('x', (5, 0, 31)),
  )
  n_schema=3, 2, 7
  t_schema=2, 0, 3
  ni=_NI(layers_len, num_seed, type_seed, (g_schema, n_schema, t_schema))
  ni.gen[0].gen._gen=bytearray(ni_str.replace(' ', '').encode())
  ni.gen[0]._update_fenotype()

  # test
  ret=core.fitness_corrector(ni, ni.gen[0].fenotype['x'])

  # results
  assert isinstance(float(ret), float)
  assert ret==pytest.approx(expected_fit)

mark__test_cr_network=pytest.mark.parametrize(
  'number_of_generations',
  [1, 2, 5]
)
@mark__test_cr_network
def test_cr_network(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  number_of_generations: int
) -> None:
  # values
  training_data=np.zeros((2, 2))
  test_data=np.zeros((2, 2))
  population_size=10
  monkeypatch.setattr(
    util,
    "get_fit_func",
    lambda *args, **kwargs: lambda x, dir: (1, ),
  )

  # test
  ret=core.cr_network(
    training_data,
    test_data,
    save_dir_path=tmp_path,
    population_size=population_size,
    number_of_generations=number_of_generations,
  )

  # results
  assert isinstance(ret, _G)
  assert ret.curr_generations==number_of_generations

mark__test_cr_network_plot=pytest.mark.parametrize(
  'number_of_generations',
  [1, 2, 5]
)
@mark__test_cr_network_plot
def test_cr_network_plot(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  number_of_generations: int,
) -> None:
  # values
  import matplotlib
  mock_plt=MockPlt()
  monkeypatch.setattr(matplotlib, "pyplot", mock_plt)
  monkeypatch.setattr(
    util,
    "get_fit_func",
    lambda *args, **kwargs: lambda x, dir: (0, ),
  )

  training_data=np.zeros((2, 2))
  test_data=np.zeros((2, 2))
  population_size=10

  # test
  ret=core.cr_network(
    training_data,
    test_data,
    save_dir_path=tmp_path,
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
    assert mock_plt.figures[k]['xlim']==(0, number_of_generations+1)
  for k in mock_plt.figures.keys():
    assert mock_plt.figures[k]['ylim']==(0, 1)

mark__test_cr_network_plot_0_in_fitnesses=pytest.mark.parametrize(
  'number_of_generations',
  [1, 2, 5]
)
@mark__test_cr_network_plot_0_in_fitnesses
def test_cr_network_plot_0_in_fitnesses(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  number_of_generations: int,
) -> None:
  # values
  import matplotlib
  mock_plt=MockPlt()
  monkeypatch.setattr(matplotlib, "pyplot", mock_plt)
  monkeypatch.setattr(
    util,
    "get_fit_func",
    lambda *args, **kwargs: lambda x, dir: (float('inf'), ),
  )

  training_data=np.zeros((2, 2))
  test_data=np.zeros((2, 2))
  population_size=10

  # test
  ret=core.cr_network(
    training_data,
    test_data,
    save_dir_path=tmp_path,
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
    assert mock_plt.figures[k]['xlim']==(0, number_of_generations+1)
  for k in mock_plt.figures.keys():
    assert mock_plt.figures[k]['ylim']==(0, .5)

mark__test_error_cr_network_0generations=pytest.mark.parametrize(
  'number_of_generations',
  [0, -1, -2]
)
@mark__test_error_cr_network_0generations
def test_error_cr_network_0generations(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
  number_of_generations: int,
) -> None:
  # values
  training_data=np.zeros((2, 2))
  test_data=np.zeros((2, 2))
  population_size=10
  monkeypatch.setattr(
    util,
    "get_fit_func",
    lambda *args, **kwargs: lambda x, dir: (1, ),
  ) # first population also has calculated fittness (before generations obj is created)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=core.cr_network(
      training_data,
      test_data,
      save_dir_path=tmp_path,
      population_size=population_size,
      number_of_generations=number_of_generations,
    )

  # results
  assert str(excinfo.value)=='max_num_gen: is too small, it should at least equal 1'
