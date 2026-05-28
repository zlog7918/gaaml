import json
import pytest
import typing as t
from pathlib import Path
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
  def _save_format(self) -> dict[str, object]:
    return {
      'name': self.__class__.__name__,
      'gen': self._gen
    }

class I_Test(I["I_Test", int, str]):
  def __init__(self, gen: str) -> None:
    super().__init__(gen)

  def mutate(self) -> None: ...

  _IT=t.TypeVar('_IT', bound="I_Test")
  @classmethod
  def get_cp(cls: type[_IT], a: _IT, b: _IT) -> int: ...
  @classmethod
  def crossover(cls: type[_IT], a: _IT, b: _IT, cp: int) -> tuple[_IT, _IT]: ...
  def _save_format(self) -> dict[str, object]:
    return {
      'name': self.__class__.__name__,
      'gen': self._gen
    }

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
  ('ind_class', 'get_file_path', 'gen'),
  [
    (BI_Test, lambda path: path/'test_model.json', 'test_text'),
    (I_Test, lambda path: path/'test_model.json', 'test_text'),
  ],
)
@mark__test_save_to
def test_save_to(
  tmp_path: Path,
  ind_class: type[BI_Test|I_Test],
  get_file_path: t.Callable[[Path], Path],
  gen: str,
) -> None:
  # values
  i=ind_class(gen)
  path=get_file_path(tmp_path)

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

mark__test_base_save_to_create_parent=pytest.mark.parametrize(
  ('ind_class', 'get_file_path', 'gen'),
  [
    (BI_Test, lambda path: path/'a'/'b'/'c'/'test_model.json', 'test_text'),
    (I_Test, lambda path: path/'a'/'b'/'c'/'test_model.json', 'test_text'),
  ],
)
@mark__test_base_save_to_create_parent
def test_base_save_to_create_parent(
  tmp_path: Path,
  ind_class: type[BI_Test|I_Test],
  get_file_path: t.Callable[[Path], Path],
  gen: str,
) -> None:
  # values
  i=ind_class(gen)
  path=get_file_path(tmp_path)

  # test
  i.save_to(path)

  # results
  assert path.exists()
  assert path.is_file()
  assert path.parent.exists()

mark__test_save_to_exists_error=pytest.mark.parametrize(
  ('ind_class', 'get_file_path', 'gen'),
  [
    (BI_Test, lambda path: path/'test_model.json', 'test_text'),
    (I_Test, lambda path: path/'test_model.json', 'test_text'),
  ],
)
@mark__test_save_to_exists_error
def test_save_to_exists_error(
  tmp_path: Path,
  ind_class: type[BI_Test|I_Test],
  get_file_path: t.Callable[[Path], Path],
  gen: str,
) -> None:
  # values
  i=ind_class(gen)
  path=get_file_path(tmp_path)

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
