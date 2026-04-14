import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import gaaml
import pytest
import numpy as np
import typing as t
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
  def ylim(self, *_: t.Any, **__: t.Any) -> None: ...
  def title(self, *_: t.Any, **__: t.Any) -> None: ...

mark__test_cr_network=pytest.mark.parametrize(
  ('number_of_generations', 'training_data', 'test_data', 'number_of_attributes'),
  [
    (1, np.zeros((2, 2)), np.zeros((2, 2)), -1),
    # (2, np.zeros((2, 2)), np.zeros((2, 2)), -1),
    # (5, np.zeros((2, 2)), np.zeros((2, 2)), -1),
    # (5, np.zeros((10, 3)), np.zeros((8, 3)), -1),
    # (5, np.zeros((10, 7)), np.zeros((8, 4)), -1),
  ]
)
@mark__test_cr_network
def test_cr_network(number_of_generations: int, training_data: np.ndarray, test_data: np.ndarray, number_of_attributes: int) -> None:
  # values
  population_size=2

  # test
  with pytest.warns() as warninfo:
    ret=gaaml.cr_network(
      training_data,
      test_data,
      number_of_attributes=number_of_attributes,
      population_size=population_size,
      number_of_generations=number_of_generations,
      max_worker_num=1,
      num_of_fittnesses_calc=1,
    )

  # results
  assert len(warninfo.list)>=2
  assert all(isinstance(warn.message, DeprecationWarning) for warn in warninfo.list)
  assert all(
    str(warn.message).find('__array__ implementation doesn\'t accept a copy keyword, so passing copy=False failed')>=0
      for warn in
    warninfo.list
  )
  assert isinstance(ret, _G)
  assert ret.curr_generations==number_of_generations

mark__test_cr_network_plot=pytest.mark.parametrize(
  ('number_of_generations', 'training_data', 'test_data', 'number_of_attributes'),
  [
    (1, np.zeros((2, 2)), np.zeros((2, 2)), -1),
    # (2, np.zeros((2, 2)), np.zeros((2, 2)), -1),
    # (5, np.zeros((2, 2)), np.zeros((2, 2)), -1),
  ]
)
@mark__test_cr_network_plot
def test_cr_network_plot(monkeypatch: pytest.MonkeyPatch, number_of_generations: int, training_data: np.ndarray, test_data: np.ndarray, number_of_attributes: int) -> None:
  # values
  import matplotlib
  mock_plt=MockPlt()
  monkeypatch.setattr(matplotlib, "pyplot", mock_plt)
  population_size=2

  # def fitness_func(lambda _f, net_ind: _f(net_ind))

  # test
  with pytest.warns() as warninfo:
    ret=gaaml.cr_network(
      training_data,
      test_data,
      number_of_attributes=number_of_attributes,
      population_size=population_size,
      number_of_generations=number_of_generations,
      plot=True,
      max_worker_num=1,
      num_of_fittnesses_calc=1,
    )

  # results
  assert len(warninfo.list)>=2
  assert all(isinstance(warn.message, DeprecationWarning) for warn in warninfo.list)
  assert all(
    str(warn.message).find('__array__ implementation doesn\'t accept a copy keyword, so passing copy=False failed')>=0
      for warn in
    warninfo.list
  )
  assert isinstance(ret, _G)
  assert ret.curr_generations==number_of_generations
  assert mock_plt.figure_calls==3
  assert len(mock_plt.figures)==3
  assert mock_plt.plot_calls==3
  assert mock_plt.show_calls==3
  for k in mock_plt.figures.keys():
    assert mock_plt.figures[k]['xlim']==(0, number_of_generations+2)

@pytest.mark.parametrize(
  'number_of_generations',
  [0, -1, -2]
)
def test_error_cr_network_0generations(number_of_generations: int) -> None:
  # values
  training_data=np.zeros((2, 2))
  test_data=np.zeros((2, 2))
  population_size=2

  # test
  with (
    pytest.raises(ValueError) as excinfo,
    pytest.warns() as warninfo,
  ):
    _=gaaml.cr_network(
      training_data,
      test_data,
      population_size=population_size,
      number_of_generations=number_of_generations,
      max_worker_num=1,
      num_of_fittnesses_calc=1,
    )

  # results
  assert len(warninfo.list)>=2
  assert all(isinstance(warn.message, DeprecationWarning) for warn in warninfo.list)
  assert all(
    str(warn.message).find('__array__ implementation doesn\'t accept a copy keyword, so passing copy=False failed')>=0
      for warn in
    warninfo.list
  )
  assert str(excinfo.value)=='max_num_gen: is too small, it should at least equal 1'
