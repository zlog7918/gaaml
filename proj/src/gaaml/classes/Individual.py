import abc
import json
import typing as t
from pathlib import Path

_T=t.TypeVar('_T')
_RT=t.TypeVar('_RT')
_BI=t.TypeVar('_BI', bound="_BaseIndividual")
_CPType=t.TypeVar('_CPType')
class _BaseIndividual(t.Generic[_BI, _CPType, _T, _RT], metaclass=abc.ABCMeta):
  _gen: _T
  def __init__(self, gen: _T, transform: t.Callable[[_T], _RT]) -> None:
    self._gen=gen
    self.__transform=transform

  @property
  def gen(self) -> _RT:
    return self.__transform(self._gen)
  
  @abc.abstractmethod
  def mutate(self) -> None: ...
  @classmethod
  @abc.abstractmethod
  def get_cp(cls: type[_BI], a: _BI, b: _BI) -> _CPType: ...
  @classmethod
  @abc.abstractmethod
  def crossover(cls: type[_BI], a: _BI, b: _BI, cp: _CPType) -> tuple[_BI, _BI]: ...

  @abc.abstractmethod
  def _save_format(self: _BI) -> dict[str, object]: ...
  def save_to(self: _BI, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'x') as model_save:
      json.dump(self._save_format(), model_save)
  @classmethod
  @abc.abstractmethod
  def _load_from_format(cls: type[_BI], saved_model: dict[str, object]) -> _BI: ...
  @classmethod
  def _load_err_raiser(cls: type[_BI]) -> t.Never:
    raise ValueError(f'Model saved is not {cls.__name__}')
  @classmethod
  def load_from(cls: type[_BI], path: Path) -> _BI:
    with open(path, 'r') as saved_model:
      saved_data=json.load(saved_model)
    if not isinstance(saved_data, dict):
      cls._load_err_raiser()
    return cls._load_from_format(saved_data)


_I=t.TypeVar('_I', bound="Individual")
class Individual(t.Generic[_I, _CPType, _T], _BaseIndividual[_I, _CPType, _T, _T], metaclass=abc.ABCMeta):
  def __init__(self, gen: _T) -> None:
    super().__init__(gen, lambda x: x)
