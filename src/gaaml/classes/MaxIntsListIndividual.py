import typing as t
from . import _utils as util
from .Individual import Individual
from .ListIndividual import (
  CPType,
  ListIndividual as _LI,
)

class MaxIntsListIndividual(Individual["MaxIntsListIndividual", CPType, _LI]):
  _MILI: t.TypeAlias="MaxIntsListIndividual"
  GenSchemaType: t.TypeAlias=tuple[tuple[int, int], util.BitSize_Min_Max]
  @t.overload
  def __init__(self, num_items: int, schema: GenSchemaType, /) -> None: ...
  @t.overload
  def __init__(self, a: _MILI, b: _MILI, /, *, cross_point: CPType) -> None: ...
  def __init__(
    self,
    a: t.Union[int, _MILI],
    b: t.Union[GenSchemaType, _MILI],
    /, *,
    cross_point: CPType|None=None,
  ) -> None:
    if isinstance(a, int) and isinstance(b, tuple):
      min_max_len, (elem_size, _min, _max)=b
      super().__init__(_LI(a, (min_max_len, elem_size)))
      self.schema=_min, _max
    else:
      if isinstance(a, int) or isinstance(b, tuple) or cross_point is None:
        raise ValueError('Illegal argument options')
      a.__same_or_err(b)
      super().__init__(_LI(a.gen, b.gen, cross_point=cross_point))
      self.schema=a.schema
    self._update_fenotype()

  def mutate(self) -> None:
    self.gen.mutate()
    self._update_fenotype()

  def _update_fenotype(self) -> None:
    self.fenotype: list[int]=self.get_fenotype(self.gen, *self.schema)

  def __same_or_err(self: _MILI, o: _MILI) -> None:
    if self.schema!=o.schema:
      raise ValueError('First and second solution do not have equal configuration')

  @staticmethod
  def get_fenotype(gen: _LI, min_v: int, max_v: int) -> list[int]:
    gen_len=len(gen.gen)
    fenotype: list[int]=[0]*(gen_len//gen.item_size)
    for i, idx in enumerate(range(0, gen_len, gen.item_size)):
      fenotype[i]=util.correct_gen_to_min_max(
        gen.gen[idx:idx+gen.item_size],
        min_v,
        max_v,
      )
    return fenotype

  @classmethod
  def get_cp(cls: type[_MILI], a: _MILI, b: _MILI) -> CPType:
    a.__same_or_err(b)
    return _LI.get_cp(a.gen, b.gen)

  @classmethod
  def crossover(cls: type[_MILI], a: _MILI, b: _MILI, cp: CPType) -> tuple[_MILI, _MILI]:
    return (cls(a, b, cross_point=cp), cls(b, a, cross_point=(cp[1], cp[0])))

  def _save_format(self) -> dict[str, object]:
    return {
      'name': self.__class__.__name__,
      'gen': self._gen._save_format(),
      'schema': list(self.schema),
    }
  @classmethod
  def __from_gen(cls: type[_MILI], gen: _LI, schema: tuple[int, int]) -> _MILI:
    i=cls.__new__(cls)
    super(cls, i).__init__(gen)
    i.schema=schema
    i._update_fenotype()
    return i
  @classmethod
  def _load_from_format(cls: type[_MILI], saved_model: dict[str, object]) -> _MILI:
    if {k for k in saved_model.keys()}!={'name', 'gen', 'schema'}:
      cls._load_err_raiser()
    if saved_model['name']!=cls.__name__:
      cls._load_err_raiser()
    if not isinstance(saved_model['gen'], dict):
      cls._load_err_raiser()
    if not isinstance(saved_model['schema'], list):
      cls._load_err_raiser()
    if len(saved_model['schema'])!=2 or any(not isinstance(e, int) for e in saved_model['schema']):
      cls._load_err_raiser()
    gen, schema=t.cast(tuple[dict[str, object], tuple[int, int]], (
      saved_model['gen'],
      tuple(saved_model['schema']),
    ))
    try:
      li=_LI._load_from_format(gen)
    except ValueError:
      cls._load_err_raiser()
    return cls.__from_gen(li, schema)
