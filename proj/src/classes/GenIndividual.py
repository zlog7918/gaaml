from __future__ import annotations
import typing as t
import random as rnd
from . import _utils as util
from .Individual import Individual

class GenIndividual(Individual[str]):
  CPType=int
  @t.overload
  def __init__(self, num_gens: int, /): ...
  @t.overload
  def __init__(self, a: GenIndividual, b: GenIndividual, /, *, cross_point: CPType): ...
  def __init__(self, a: int|GenIndividual, b: GenIndividual|None=None, /, *, cross_point: CPType|None=None):
    if isinstance(a, int):
      super().__init__(util.int_to_bin(rnd.getrandbits(a), a))
      return
    if b is None or cross_point is None:
      raise Exception('Illegal argument options')
    if len(a.gen)!=len(b.gen):
      raise Exception('First and second solution are not equal in size')
    if cross_point<=0 or cross_point>=len(b.gen):
      raise Exception('Cross point is out side of solution')
    super().__init__(a.gen[:cross_point]+b.gen[cross_point:])

  def mutate(self) -> None:
    i=rnd.randint(0, len(self.gen)-1)
    # self.gen[i]=str(1-int(self.gen[i]))
    bit='0' if self.gen[i]=='1' else '1'
    self.gen=f'{self.gen[:i]}{bit}{self.gen[i+1:]}'

  _GI=t.TypeVar('_GI', bound=GenIndividual)
  @classmethod
  def get_cp(cls: type[_GI], a: _GI, _: _GI) -> CPType:
    return rnd.randint(0, len(a.gen))

  @classmethod
  def crossover(cls: type[_GI], a: _GI, b: _GI, cp: CPType) -> tuple[_GI, _GI]:
    return (cls(a, b, cross_point=cp), cls(b, a, cross_point=cp))
