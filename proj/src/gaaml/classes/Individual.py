import abc
import typing as t

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

_I=t.TypeVar('_I', bound="Individual")
class Individual(t.Generic[_I, _CPType, _T], _BaseIndividual[_I, _CPType, _T, _T], metaclass=abc.ABCMeta):
  def __init__(self, gen: _T) -> None:
    super().__init__(gen, lambda x: x)
