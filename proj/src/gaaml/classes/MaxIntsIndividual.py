import typing as t
from . import _utils as util
from .Individual import Individual
from .GenIndividual import (
  CPType,
  GenIndividual as _GI,
)

class MaxIntsIndividual(Individual["MaxIntsIndividual", CPType, _GI]):
  _MII: t.TypeAlias="MaxIntsIndividual"
  EntryType: t.TypeAlias=tuple[str, util.BitSize_Min_Max]
  GenSchemaType: t.TypeAlias=tuple[EntryType, ...]
  __schema: GenSchemaType
  @t.overload
  def __init__(self, gen_schema: GenSchemaType, /) -> None: ...
  @t.overload
  def __init__(self, a: _MII, b: _MII, /, *, cross_point: CPType) -> None: ...
  def __init__(self, a: "GenSchemaType|_MII", b: "_MII|None"=None, /, *, cross_point: CPType|None=None) -> None:
    if isinstance(a, tuple):
      n=sum(l for _,(l,_,_) in a)
      names=tuple(n for n,(_,_,_) in a)
      if len({n for n in names})!=len(names):
        raise ValueError('Names can not collide')
      super().__init__(_GI(n))
      self.__schema=a
    else:
      if b is None or cross_point is None:
        raise ValueError('Illegal argument options')
      a.__same_or_err(b)
      super().__init__(_GI(a.gen, b.gen, cross_point=cross_point))
      self.__schema=a.__schema
    self._update_fenotype()

  def mutate(self) -> None:
    self.gen.mutate()
    self._update_fenotype()

  def _update_fenotype(self) -> None:
    self.fenotype: dict[str, int]=self.get_fenotype(self.gen, self.__schema)

  @staticmethod
  def get_fenotype(gen: _GI, schema: GenSchemaType) -> dict[str, int]:
    fenotype: dict[str, int]={}
    g_idx=0
    for g_name, (l, min_v, max_v) in schema:
      fenotype[g_name]=util.correct_gen_to_min_max(
        gen.gen[g_idx:g_idx+l],
        min_v,
        max_v,
      )
      g_idx+=l
    return fenotype

  def __same_or_err(self: _MII, o: _MII) -> None:
    if self.__schema!=o.__schema:
      raise ValueError('First and second solution do not have equal configuration')

  @classmethod
  def get_cp(cls: type[_MII], a: _MII, b: _MII) -> CPType:
    a.__same_or_err(b)
    return _GI.get_cp(a.gen, b.gen)

  @classmethod
  def crossover(cls: type[_MII], a: _MII, b: _MII, cp: CPType) -> tuple[_MII, _MII]:
    return (cls(a, b, cross_point=cp), cls(b, a, cross_point=cp))

  def _save_format(self) -> dict[str, object]:
    return {
      'name': self.__class__.__name__,
      'gen': self._gen._save_format()
    }
