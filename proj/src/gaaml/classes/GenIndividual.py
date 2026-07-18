import typing as t
import random as rnd
from . import _utils as util
from .Individual import Individual

CPType: t.TypeAlias=int
class GenIndividual(Individual["GenIndividual", CPType, bytearray]):
  _GI: t.TypeAlias="GenIndividual"
  @t.overload
  def __init__(self, num_gens: int, /) -> None: ...
  @t.overload
  def __init__(self: _GI, a: _GI, b: _GI, /, *, cross_point: CPType) -> None: ...
  def __init__(self: _GI, a: "int|_GI", b: "_GI|None"=None, /, *, cross_point: CPType|None=None) -> None:
    if isinstance(a, int):
      super().__init__(util.int_to_bin(rnd.getrandbits(a), a))
      return
    if b is None or cross_point is None:
      raise ValueError('Illegal argument options')
    a.__same_or_err(b)
    if cross_point<0 or cross_point>len(b.gen):
      raise ValueError('Cross point is outside of solution')
    super().__init__(a.gen[:cross_point]+b.gen[cross_point:])

  def mutate(self) -> None:
    i=rnd.randint(0, len(self.gen)-1)
    self.gen[i]=util.ord0 if self.gen[i]==util.ord1 else util.ord1
    # self.gen[i]=str(1-int(self.gen[i]))
    # bit='0' if self.gen[i]=='1' else '1'
    # self.gen=f'{self.gen[:i]}{bit}{self.gen[i+1:]}'

  def __same_or_err(self: _GI, o: _GI) -> None:
    if len(self.gen)!=len(o.gen):
      raise ValueError('First and second solution are not equal in size')

  @classmethod
  def get_cp(cls: type[_GI], a: _GI, b: _GI) -> CPType:
    a.__same_or_err(b)
    return rnd.randint(0, len(a.gen))

  @classmethod
  def crossover(cls: type[_GI], a: _GI, b: _GI, cp: CPType) -> tuple[_GI, _GI]:
    return (cls(a, b, cross_point=cp), cls(b, a, cross_point=cp))

  def _save_format(self) -> dict[str, object]:
    return {
      'name': self.__class__.__name__,
      'gen': self._gen.decode(),
    }
  @classmethod
  def __from_gen(cls: type[_GI], gen: bytearray) -> _GI:
    i=cls.__new__(cls)
    super(cls, i).__init__(gen)
    return i
  @classmethod
  def _load_from_format(cls: type[_GI], saved_model: dict[str, object]) -> _GI:
    if {k for k in saved_model.keys()}!={'name', 'gen'}:
      cls._load_err_raiser()
    if saved_model['name']!=cls.__name__:
      cls._load_err_raiser()
    if not isinstance(saved_model['gen'], str):
      cls._load_err_raiser()
    if len(saved_model['gen'])==0:
      cls._load_err_raiser()
    if not set(saved_model['gen']).issubset({'0', '1'}):
      cls._load_err_raiser()
    return cls.__from_gen(bytearray(saved_model['gen'].encode()))
