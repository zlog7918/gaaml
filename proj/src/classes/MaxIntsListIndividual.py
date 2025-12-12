from __future__ import annotations
import typing as t
from . import _utils as util
from .Individual import Individual
from .ListIndividual import ListIndividual

class MaxIntsListIndividual(Individual[ListIndividual]):
  GenShemaType=tuple[int, int, int]
  CPType=ListIndividual.CPType
  @t.overload
  def __init__(self, num_items: int, shema: GenShemaType, /): ...
  @t.overload
  def __init__(self, a: MaxIntsListIndividual, b: MaxIntsListIndividual, /, *, cross_point: CPType): ...
  def __init__(self, a: int|MaxIntsListIndividual, b: GenShemaType|MaxIntsListIndividual, /, *, cross_point: CPType|None=None):
    if isinstance(a, int) and isinstance(b, tuple):
      super().__init__(ListIndividual(a, b[0]))
      self.shema=b
    else:
      if isinstance(a, int) or isinstance(b, tuple) or cross_point is None:
        raise Exception('Illegal argument options')
      if a.shema!=b.shema:
        raise Exception('First and second solution do not have equal configuration')
      super().__init__(ListIndividual(a.gen, b.gen, cross_point=cross_point))
      self.shema=a.shema
    self._update_fenotype()

  def mutate(self) -> None:
    self.gen.mutate()
    self._update_fenotype()

  def _update_fenotype(self) -> None:
    self.fenotype: list[int]=[]
    l, min_v, max_v=self.shema
    for idx in range(0, len(self.gen.gen), l):
      self.fenotype.append(
        util.correct_gen_to_min_max(
          self.gen.gen[idx:idx+l],
          min_v,
          max_v,
        )
      )

  _VGI=t.TypeVar('_VGI', bound=MaxIntsListIndividual)
  @classmethod
  def get_cp(cls: type[_VGI], a: _VGI, b: _VGI) -> CPType:
    return ListIndividual.get_cp(a.gen, b.gen)

  @classmethod
  def crossover(cls: type[_VGI], a: _VGI, b: _VGI, cp: CPType) -> tuple[_VGI, _VGI]:
    return (cls(a, b, cross_point=cp), cls(b, a, cross_point=(cp[1], cp[0])))
