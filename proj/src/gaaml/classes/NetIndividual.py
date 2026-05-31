import typing as t
import random as rnd
from . import _utils as util
from .Individual import Individual
from .MaxIntsIndividual import (
  CPType as MII_CPType,
  MaxIntsIndividual as _MII,
)
from .MaxIntsListIndividual import (
  MaxIntsListIndividual as _MILI,
  CPType as MILI_CPType,
)

CPType: t.TypeAlias=tuple[MII_CPType, MILI_CPType, MILI_CPType]
class NetIndividual(Individual["NetIndividual", CPType, tuple[_MII, _MILI, _MILI]]):
  _NI: t.TypeAlias="NetIndividual"
  GenSchemaType: t.TypeAlias=tuple[_MII.GenSchemaType, util.BitSize_Min_Max, util.BitSize_Min_Max]
  layers_len_name: str
  num_seed_name: str
  type_seed_name: str
  type_len_modifier: t.Callable[[int], int]=staticmethod(lambda x: x+1)
  @t.overload
  def __init__(
    self,
    layers_len: _MII.EntryType,
    num_seed: _MII.EntryType,
    type_seed: _MII.EntryType,
    gen_schema: GenSchemaType,
    /,
  ) -> None: ...
  @t.overload
  def __init__(
    self,
    a: _NI,
    b: _NI,
    /, *,
    cross_point: CPType
  ) -> None: ...
  def __init__(
    self,
    a: "_MII.EntryType|_NI",
    b: "_MII.EntryType|_NI",
    type_seed: _MII.EntryType|None=None,
    gen_schema: GenSchemaType|None=None,
    /, *,
    cross_point: CPType|None=None,
  ) -> None:
    if (
      isinstance(a, tuple)
      and isinstance(b, tuple)
      and type_seed is not None
      and gen_schema is not None
    ):
      layers_len_name, (_, layers_len_min, layers_len_max)=a
      g_schema, l_schema, t_schema=gen_schema
      g_schema=(a, b, type_seed, *g_schema)
      g=_MII(g_schema)
      list_len=g.fenotype[layers_len_name]
      l=_MILI(
        list_len,
        ((
          layers_len_min,
          layers_len_max+1,
        ), l_schema),
      )
      t=_MILI(
        self.type_len_modifier(list_len),
        ((
          self.type_len_modifier(layers_len_min),
          self.type_len_modifier(layers_len_max+1),
        ), t_schema),
      )
      super().__init__((g, l, t))
      self.layers_len_name=layers_len_name
      self.num_seed_name, _=b
      self.type_seed_name, _=type_seed
      return
    if isinstance(a, tuple) or isinstance(b, tuple) or cross_point is None:
      raise ValueError('Illegal argument options')
    (
      (ag, al, at),
      (bg, bl, bt),
      (cg, cl, ct),
    )=a.gen, b.gen, cross_point
    g=_MII(ag, bg, cross_point=cg)
    l=_MILI(al, bl, cross_point=cl)
    t=_MILI(at, bt, cross_point=ct)
    super().__init__((g, l, t))
    self.layers_len_name=a.layers_len_name
    self.num_seed_name=a.num_seed_name
    self.type_seed_name=a.type_seed_name
    self._update()

  def mutate(self) -> None:
    n=len(self.gen)
    sampled=rnd.sample(
      range(n),
      util.randint(1, n)
    )
    for i in sampled:
      self.gen[i].mutate()
    self._update()

  def _update(self) -> None:
    g, l, t=self.gen
    list_len=g.fenotype[self.layers_len_name]
    to_add: list[tuple[_MILI, int, int]]=[]
    if len(l.fenotype)<list_len:
      to_add.append((l, list_len-len(l.fenotype), g.fenotype[self.num_seed_name]))
    else:
      l.fenotype=l.fenotype[:list_len]
    list_len=self.type_len_modifier(list_len)
    if len(t.fenotype)<list_len:
      to_add.append((t, list_len-len(t.fenotype), g.fenotype[self.type_seed_name]))
    else:
      t.fenotype=t.fenotype[:list_len]

    for to_a_l, to_a_n, to_a_seed in to_add:
      r=rnd.Random(to_a_seed)
      nums=(
          r.randint(*to_a_l.schema)
        for _ in
          range(to_a_n)
      )
      to_a_l.fenotype.extend(nums)

  @classmethod
  def get_cp(cls: type[_NI], a: _NI, b: _NI) -> CPType:
    ag, al, at=a.gen
    bg, bl, bt=b.gen
    return (
      _MII.get_cp(ag, bg),
      _MILI.get_cp(al, bl),
      _MILI.get_cp(at, bt),
    )

  @classmethod
  def crossover(cls: type[_NI], a: _NI, b: _NI, cp: CPType) -> tuple[_NI, _NI]:
    c0, (ac1, bc1), (ac2, bc2)=cp
    return (
      cls(a, b, cross_point=cp),
      cls(b, a, cross_point=(c0, (bc1, ac1), (bc2, ac2))),
    )

  def _save_format(self) -> dict[str, object]:
    gen_g, gen_l, gen_t=self._gen
    return {
      'name': self.__class__.__name__,
      'gen': {
        'g': gen_g._save_format(),
        'l': gen_l._save_format(),
        't': gen_t._save_format(),
      }
    }
