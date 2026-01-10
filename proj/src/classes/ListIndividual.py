import typing as t
import random as rnd
from . import _utils as util
from .Individual import Individual

class ListIndividual(Individual[bytearray]):
  GenSchemaType: t.TypeAlias=tuple[tuple[int, int], int]
  CPType: t.TypeAlias=tuple[int, int]
  item_size: int
  min_elem_len: int
  max_bit_len: int
  @t.overload
  def __init__(self, num_items: int, schema: GenSchemaType, /): ...
  @t.overload
  def __init__(self, a: "ListIndividual", b: "ListIndividual", /, *, cross_point: CPType): ...
  def __init__(self, a: "int|ListIndividual", b: "GenSchemaType|ListIndividual", /, *, cross_point: CPType|None=None):
    if isinstance(a, int) and isinstance(b, tuple):
      (_min, _max), it_size=b
      if a>_max or a<_min:
        raise ValueError('List size out of allowed range')
      l=a*it_size
      super().__init__(util.int_to_bin(rnd.getrandbits(l), l))
      self.min_elem_len=_min
      self.max_bit_len=it_size*_max
      self.item_size=it_size
      return
    if isinstance(a, int) or isinstance(b, tuple) or cross_point is None:
      raise TypeError('Illegal argument options')
    a._same_or_err(b)
    if any(
      cp<0 or cp>len(gen)
        for gen, cp in
      zip((a.gen, b.gen), cross_point)
    ):
      raise ValueError('Cross point is out side of solution')
    super().__init__((a.gen[:cross_point[0]]+b.gen[cross_point[1]:])[:a.max_bit_len])
    self.item_size=a.item_size
    self.max_bit_len=a.max_bit_len
    self.min_elem_len=a.min_elem_len

  def mutate(self) -> None:
    if self.gen:
      i=util.randint(0, len(self.gen)-1)
      # self.gen[i]=1-self.gen[i]
      self.gen[i]=util.ord0 if self.gen[i]==util.ord1 else util.ord1
      # bit='0' if self.gen[i]=='1' else '1'
      # self.gen=f'{self.gen[:i]}{bit}{self.gen[i+1:]}'

  _VGI=t.TypeVar('_VGI', bound="ListIndividual")
  def _same_or_err(self: _VGI, o: _VGI) -> None:
    if self.item_size!=o.item_size:
      raise TypeError('First and second solution do not have equal gen size')
    if self.max_bit_len!=o.max_bit_len:
      raise TypeError('First and second solution do not have equal max list size')
    if self.min_elem_len!=o.min_elem_len:
      raise TypeError('First and second solution do not have equal min list size')

  @classmethod
  def get_cp(cls: type[_VGI], a: _VGI, b: _VGI) -> CPType:
    a._same_or_err(b)
    l_min, es=a.min_elem_len, a.item_size
    la=len(a.gen)
    ira=util.randint(0, la)
    lb=ira%es
    is_r=int(lb>0)
    lpca=ira//es
    lkcb=la//es-lpca
    nrb=len(b.gen)//es
    irb=util.randint(
      0 if lkcb>=l_min else (l_min-lkcb),
      nrb-(is_r if lpca+is_r>=l_min else (l_min-lpca)),
    )
    irb=irb*es+lb
    return ira, irb

  @classmethod
  def crossover(cls: type[_VGI], a: _VGI, b: _VGI, cp: CPType) -> tuple[_VGI, _VGI]:
    return (cls(a, b, cross_point=cp), cls(b, a, cross_point=(cp[1], cp[0])))
