import json
import pytest
import typing as t
from pathlib import Path
from gaaml.classes.Individual import (
  Individual as I,
  _BaseIndividual as BI,
)

class BI_Test(BI["BI_Test", int, str, list[str]]):
  _BIT: t.TypeAlias="BI_Test"
  def __init__(self, gen: str) -> None:
    super().__init__(gen, lambda x: list(x))
  def mutate(self) -> None: ...

  @classmethod
  def get_cp(cls: type[_BIT], a: _BIT, b: _BIT) -> int: ...
  @classmethod
  def crossover(cls: type[_BIT], a: _BIT, b: _BIT, cp: int) -> tuple[_BIT, _BIT]: ...
  def _save_format(self) -> dict[str, object]:
    return {
      'name': self.__class__.__name__,
      'gen': self._gen
    }
  @classmethod
  def _load_from_format(cls: type[_BIT], saved_model: dict[str, object]) -> _BIT:
    ind=cls(t.cast(str, saved_model['gen']))
    ind.from_model=saved_model # type: ignore
    return ind

class I_Test(I["I_Test", int, str]):
  _IT: t.TypeAlias="I_Test"
  def __init__(self, gen: str) -> None:
    super().__init__(gen)
  def mutate(self) -> None: ...

  @classmethod
  def get_cp(cls: type[_IT], a: _IT, b: _IT) -> int: ...
  @classmethod
  def crossover(cls: type[_IT], a: _IT, b: _IT, cp: int) -> tuple[_IT, _IT]: ...
  def _save_format(self) -> dict[str, object]:
    return {
      'name': self.__class__.__name__,
      'gen': self._gen
    }
  @classmethod
  def _load_from_format(cls: type[_IT], saved_model: dict[str, object]) -> _IT:
    ind=cls(t.cast(str, saved_model['gen']))
    ind.from_model=saved_model # type: ignore
    return ind

mark__test_create=pytest.mark.parametrize(
  ('ind_class', 'gen', 'expected_gen'),
  [
    (BI_Test, 'test_text', ['t', 'e', 's', 't', '_', 't', 'e', 'x', 't']),
    (I_Test, 'test_text', 'test_text'),
  ],
)
@mark__test_create
def test_create(ind_class: type[BI_Test|I_Test], gen: str, expected_gen: object) -> None:
  # values ^

  # test
  i=ind_class(gen)

  # results
  assert isinstance(i._gen, str)
  assert isinstance(i.gen, type(expected_gen))
  assert i._gen==gen
  assert i.gen==expected_gen

mark__test_save_to=pytest.mark.parametrize(
  ('ind_class', 'gen'),
  [
    (BI_Test, 'test_text'),
    (I_Test, 'test_text'),
  ],
)
@mark__test_save_to
def test_save_to(
  tmp_path: Path,
  ind_class: type[BI_Test|I_Test],
  gen: str,
) -> None:
  # values
  i=ind_class(gen)
  path=tmp_path/'test_model.json'

  # test
  i.save_to(path)

  # results
  assert path.exists()
  assert path.is_file()

  with open(path) as saved_model:
    data=json.load(saved_model)

  assert data=={
    'name': ind_class.__name__,
    'gen': gen,
  }

mark__test_save_to_create_parent=pytest.mark.parametrize(
  ('ind_class', 'gen'),
  [
    (BI_Test, 'test_text'),
    (I_Test, 'test_text'),
  ],
)
@mark__test_save_to_create_parent
def test_save_to_create_parent(
  tmp_path: Path,
  ind_class: type[BI_Test|I_Test],
  gen: str,
) -> None:
  # values
  i=ind_class(gen)
  path=tmp_path/'a'/'b'/'c'/'test_model.json'

  # test
  i.save_to(path)

  # results
  assert path.exists()
  assert path.is_file()
  assert path.parent.exists()

mark__test_load_from=pytest.mark.parametrize(
  ('ind_class', 'gen'),
  [
    (BI_Test, 'test_text'),
    (I_Test, 'test_text'),
  ],
)
@mark__test_load_from
def test_load_from(
  tmp_path: Path,
  ind_class: type[BI_Test|I_Test],
  gen: str,
) -> None:
  # values
  path=tmp_path/'test_model.json'
  oryg_i=ind_class(gen)

  # setup
  oryg_i.save_to(path)

  # test
  i=ind_class.load_from(path)

  # results
  assert isinstance(i, ind_class)
  assert i.gen==oryg_i.gen
  assert i._gen==oryg_i._gen

mark__test_load_from_calls_load_from_format=pytest.mark.parametrize(
  ('ind_class', 'gen'),
  [
    (BI_Test, 'test_text'),
    (I_Test, 'test_text'),
  ],
)
@mark__test_load_from_calls_load_from_format
def test_load_from_calls_load_from_format(
  tmp_path: Path,
  ind_class: type[BI_Test|I_Test],
  gen: str,
) -> None:
  # values
  path=tmp_path/'test_model.json'
  saved_data={
    'name': ind_class.__name__,
    'gen': gen,
  }
  with open(path, 'w') as saved_model:
    json.dump(saved_data, saved_model)

  # test
  i=ind_class.load_from(path)

  # results
  assert i.from_model==saved_data # type: ignore

mark__test_error_save_to_exists=pytest.mark.parametrize(
  ('ind_class', 'gen'),
  [
    (BI_Test, 'test_text'),
    (I_Test, 'test_text'),
  ],
)
@mark__test_error_save_to_exists
def test_error_save_to_exists(
  tmp_path: Path,
  ind_class: type[BI_Test|I_Test],
  gen: str,
) -> None:
  # values
  i=ind_class(gen)
  path=tmp_path/'test_model.json'

  # setup
  path.touch()

  # test/results
  with pytest.raises(FileExistsError):
    i.save_to(path)

mark__test_error_change_gen=pytest.mark.parametrize(
  ('ind_class', 'gen', 'gen_to_set'),
  [
    (BI_Test, 'test_text', 'dummy_text'),
    (I_Test, 'test_text', 'dummy_text'),
  ],
)
@mark__test_error_change_gen
def test_error_change_gen(
  ind_class: type[BI_Test|I_Test],
  gen: str,
  gen_to_set: str,
) -> None:
  # values
  i=ind_class(gen)

  # test
  with pytest.raises(AttributeError) as excinfo:
    i.gen=gen_to_set # type: ignore

  # results
  assert str(excinfo.value) in {
    'can\'t set attribute \'gen\'',
    f'property \'gen\' of \'{ind_class.__name__}\' object has no setter',
  }
