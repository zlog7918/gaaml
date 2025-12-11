from __future__ import annotations
import typing as t
from . import utils as util
from .Individual import Individual
from .GenIndividual import GenIndividual

class MaxIntsIndividual(Individual[GenIndividual]):
  GenShemaType=tuple[tuple[str, tuple[int, int, int]], ...]
  CPType=GenIndividual.CPType
  @t.overload
  def __init__(self, gen_shema: GenShemaType, /): ...
  @t.overload
  def __init__(self, a: MaxIntsIndividual, b: MaxIntsIndividual, /, *, cross_point: CPType): ...
  def __init__(self, a: GenShemaType|MaxIntsIndividual, b: MaxIntsIndividual|None=None, /, *, cross_point: CPType|None=None):
    if isinstance(a, tuple):
      n=sum([l for _,(l,_,_) in a])
      super().__init__(GenIndividual(n))
      self.shema=a
    else:
      if b is None or cross_point is None:
        raise Exception('Illegal argument options')
      super().__init__(GenIndividual(a.gen, b.gen, cross_point=cross_point))
      self.shema=a.shema
    self._update_fenotype()

  def _update_fenotype(self) -> None:
    self.fenotype: dict[str, int]={}
    g_idx=0
    for g_name, (l, min_v, max_v) in self.shema:
      self.fenotype[g_name]=util.correct_gen_to_min_max(
        self.gen.gen[g_idx:g_idx+l],
        min_v,
        max_v,
      )
      g_idx+=l

  def mutate(self) -> None:
    self.gen.mutate()
    self._update_fenotype()

  _MII=t.TypeVar('_MII', bound=MaxIntsIndividual)
  @classmethod
  def get_cp(cls: type[_MII], a: _MII, b: _MII) -> CPType:
    return GenIndividual.get_cp(a.gen, b.gen)

  @classmethod
  def crossover(cls: type[_MII], a: _MII, b: _MII, cp: CPType) -> tuple[_MII, _MII]:
    return (cls(a, b, cross_point=cp), cls(b, a, cross_point=cp))
