import pytest
import typing as t
from gaaml.classes.Individual import (
  Individual as I,
  _BaseIndividual as BI,
)

class BI_Test(BI["BI_Test", int, str, list[str]]):
  def __init__(self, gen: str) -> None:
    super().__init__(gen, lambda x: list(x))

  def mutate(self) -> None: ...

  _IT=t.TypeVar('_IT', bound="BI_Test")
  @classmethod
  def get_cp(cls: type[_IT], a: _IT, b: _IT) -> int: ...
  @classmethod
  def crossover(cls: type[_IT], a: _IT, b: _IT, cp: int) -> tuple[_IT, _IT]: ...

class I_Test(I["I_Test", int, str]):
  def __init__(self, gen: str) -> None:
    super().__init__(gen)

  def mutate(self) -> None: ...

  _IT=t.TypeVar('_IT', bound="I_Test")
  @classmethod
  def get_cp(cls: type[_IT], a: _IT, b: _IT) -> int: ...
  @classmethod
  def crossover(cls: type[_IT], a: _IT, b: _IT, cp: int) -> tuple[_IT, _IT]: ...

def test_base_create() -> None:
  # values
  gen='test_text'
  expected_gen=['t', 'e', 's', 't', '_', 't', 'e', 'x', 't']

  # test
  i=BI_Test(gen)

  # results
  assert isinstance(i._gen, str)
  assert isinstance(i.gen, list)
  assert i._gen==gen
  assert i.gen==expected_gen

def test_base_error_change_gen() -> None:
  # values
  dummy_gen='test_text'
  i=BI_Test(dummy_gen)

  # test
  with pytest.raises(AttributeError) as excinfo:
    i.gen='dummy_text' # type: ignore

  # results
  assert str(excinfo.value) in {'can\'t set attribute \'gen\'', 'property \'gen\' of \'BI_Test\' object has no setter'}

def test_create() -> None:
  # values
  dummy_gen='test_text'

  # test
  i=I_Test(dummy_gen)

  # results
  assert isinstance(i._gen, str)
  assert isinstance(i.gen, str)
  assert i.gen==dummy_gen

def test_error_change_gen() -> None:
  # values
  dummy_gen='test_text'
  i=I_Test(dummy_gen)

  # test
  with pytest.raises(AttributeError) as excinfo:
    i.gen='dummy_text' # type: ignore

  # results
  assert str(excinfo.value) in {'can\'t set attribute \'gen\'', 'property \'gen\' of \'I_Test\' object has no setter'}
