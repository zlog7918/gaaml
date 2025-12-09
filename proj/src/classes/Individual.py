from __future__ import annotations
import typing as t

_T=t.TypeVar('_T')
class Individual(t.Generic[_T]):
  gen: _T
  def __init__(self, gen: _T):
    self.gen=gen
  def mutate(self) -> None: ...
