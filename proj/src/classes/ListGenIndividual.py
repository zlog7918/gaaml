from __future__ import annotations
import typing as t
import random as rnd
from . import utils as util
from .Individual import Individual

class ListGenIndividual(Individual[str]):
  gen_size: int
  @t.overload
  def __init__(self, num_gens: int, gen_size: int, /): ...
  @t.overload
  def __init__(self, a: ListGenIndividual, b: ListGenIndividual, /, cross_point: tuple[int, int]): ...
  def __init__(self, a: int|ListGenIndividual, b: int|ListGenIndividual, /, cross_point: tuple[int, int]|None=None):
    if isinstance(a, int) and isinstance(b, int):
      super().__init__(util.int_to_bin(rnd.getrandbits(a), a))
      self.gen_size=b
      return
    if isinstance(a, int) or isinstance(b, int) or cross_point is None:
      raise Exception('Illegal argument options')
    if a.gen_size!=b.gen_size:
      raise Exception('First and second solution do not have equal gen size')
    if any(cp<0 or cp>len(gen) for gen, cp in zip((a.gen, b.gen), cross_point)):
      raise Exception('Cross point is out side of solution')
    super().__init__(a.gen[:cross_point[0]]+b.gen[cross_point[1]:])
    self.gen_size=a.gen_size

  def mutate(self) -> None:
    i=rnd.randint(0, len(self.gen)-1)
    # self.gen[i]=str(1-int(self.gen[i]))
    bit='0' if self.gen[i]=='1' else '1'
    self.gen=f'{self.gen[:i]}{bit}{self.gen[i+1:]}'

  _VGI=t.TypeVar('_VGI', bound=ListGenIndividual)
  @classmethod
  def get_cp(cls: type[_VGI], a: _VGI, b: _VGI) -> tuple[int, int]:
    if a.gen_size!=b.gen_size:
      raise Exception('First and second solution do not have equal gen size')
    ia=rnd.randint(0, len(a.gen))
    r=ia%a.gen_size
    lb=len(b.gen)//b.gen_size
    ib=rnd.randint(0, lb if r==0 else lb-1)
    ib=ib*b.gen_size+r
    return ia, ib

  @classmethod
  def crossover(cls: type[_VGI], a: _VGI, b: _VGI, cp: tuple[int, int]) -> tuple[_VGI, _VGI]:
    return (cls(a, b, cross_point=cp), cls(b, a, cross_point=(cp[1], cp[0])))
