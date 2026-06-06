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
  @t.overload
  def __init__(self, gen_schema: GenSchemaType, /) -> None: ...
  @t.overload
  def __init__(self, a: _MII, b: _MII, /, *, cross_point: CPType) -> None: ...
  def __init__(
    self,
    a: t.Union[GenSchemaType, _MII],
    b: t.Union[_MII, None]=None,
    /, *,
    cross_point: CPType|None=None,
  ) -> None:
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
      'gen': self._gen._save_format(),
      'schema': [
        [s, b, _min, _max] for s, (b, _min, _max) in self.__schema
      ],
    }
  @classmethod
  def __from_gen(cls: type[_MII], gen: _GI, schema: GenSchemaType) -> _MII:
    i=cls.__new__(cls)
    super(cls, i).__init__(gen)
    i.__schema=schema
    i._update_fenotype()
    return i
  @classmethod
  def _load_from_format(cls: type[_MII], saved_model: dict[str, object]) -> _MII:
    if {k for k in saved_model.keys()}!={'name', 'gen', 'schema'}:
      raise ValueError(f'Model saved is not {cls.__name__}')
    if saved_model['name']!=cls.__name__:
      raise ValueError(f'Model saved is not {cls.__name__}')
    if any(not isinstance(saved_model[k], _type) for k, _type in (
      ('gen', dict),
      ('schema', list),
    )):
      raise ValueError(f'Model saved is not {cls.__name__}')
    exp_entry_type=(str, int, int, int)
    if any(
      (
        not isinstance(entry, list)
        or len(entry)!=len(exp_entry_type)
        or any(not isinstance(e, _type) for e, _type in zip(entry, exp_entry_type))
      )
        for entry in
      t.cast(list, saved_model['schema'])
    ):
      raise ValueError(f'Model saved is not {cls.__name__}')
    gen, schema=t.cast(tuple[dict[str, object], MaxIntsIndividual.GenSchemaType], (
      saved_model['gen'],
      tuple((entry[0], tuple(entry[1:])) for entry in t.cast(list[list[object]], saved_model['schema']))
    ))
    names=[name for name, _ in schema]
    if len(set(names))!=len(names):
      raise ValueError(f'Model saved is not {cls.__name__}')
    try:
      gi=_GI._load_from_format(gen)
    except ValueError:
      raise ValueError(f'Model saved is not {cls.__name__}')
    return cls.__from_gen(gi, schema)
