import abc
import typing as t

_T=t.TypeVar('_T')
class Individual(t.Generic[_T], metaclass=abc.ABCMeta):
  gen: _T
  def __init__(self, gen: _T):
    self.gen=gen
  @abc.abstractmethod
  def mutate(self) -> None: ...
