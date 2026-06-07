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
      g_gen=_MII(g_schema)
      list_len=g_gen.fenotype[layers_len_name]
      l_gen=_MILI(
        list_len,
        ((
          layers_len_min,
          layers_len_max+1,
        ), l_schema),
      )
      t_gen=_MILI(
        self.type_len_modifier(list_len),
        ((
          self.type_len_modifier(layers_len_min),
          self.type_len_modifier(layers_len_max+1),
        ), t_schema),
      )
      super().__init__((g_gen, l_gen, t_gen))
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
    g_gen=_MII(ag, bg, cross_point=cg)
    l_gen=_MILI(al, bl, cross_point=cl)
    t_gen=_MILI(at, bt, cross_point=ct)
    super().__init__((g_gen, l_gen, t_gen))
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
    g_gen, l_gen, t_gen=self.gen
    list_len=g_gen.fenotype[self.layers_len_name]
    to_add: list[tuple[_MILI, int, int]]=[]
    if len(l_gen.fenotype)<list_len:
      to_add.append((l_gen, list_len-len(l_gen.fenotype), g_gen.fenotype[self.num_seed_name]))
    else:
      l_gen.fenotype=l_gen.fenotype[:list_len]
    list_len=self.type_len_modifier(list_len)
    if len(t_gen.fenotype)<list_len:
      to_add.append((t_gen, list_len-len(t_gen.fenotype), g_gen.fenotype[self.type_seed_name]))
    else:
      t_gen.fenotype=t_gen.fenotype[:list_len]

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
    cg, (acl, bcl), (act, bct)=cp
    return (
      cls(a, b, cross_point=cp),
      cls(b, a, cross_point=(cg, (bcl, acl), (bct, act))),
    )

  def _save_format(self) -> dict[str, object]:
    gen_g, gen_l, gen_t=self._gen
    return {
      'name': self.__class__.__name__,
      'gen': {
        'g': gen_g._save_format(),
        'l': gen_l._save_format(),
        't': gen_t._save_format(),
      },
      'layers_len_name': self.layers_len_name,
      'num_seed_name': self.num_seed_name,
      'type_seed_name': self.type_seed_name,
    }
  @classmethod
  def __from_gen(
    cls: type[_NI],
    gen: tuple[_MII, _MILI, _MILI],
    layers_len_name: str,
    num_seed_name: str,
    type_seed_name: str,
  ) -> _NI:
    i=cls.__new__(cls)
    super(cls, i).__init__(gen)
    i.layers_len_name=layers_len_name
    i.num_seed_name=num_seed_name
    i.type_seed_name=type_seed_name
    i._update()
    return i

  @classmethod
  def _load_from_format(cls: type[_NI], saved_model: dict[str, object]) -> _NI:
    if set(saved_model.keys())!={
      'name',
      'gen',
      'layers_len_name',
      'num_seed_name',
      'type_seed_name',
    }:
      cls._load_err_raiser()
    if saved_model['name']!=cls.__name__:
      cls._load_err_raiser()

    if any(not isinstance(saved_model[k], type) for k, type in (
      ('gen', dict),
      ('layers_len_name', str),
      ('num_seed_name', str),
      ('type_seed_name', str),
    )):
      cls._load_err_raiser()
    if set(t.cast(dict, saved_model['gen']).keys())!={'g', 'l', 't'}:
      cls._load_err_raiser()
    (
      gen,
      layers_len_name,
      num_seed_name,
      type_seed_name,
    )=t.cast(
      tuple[dict[str, dict[str, object]], str, str, str],
      map(saved_model.get, (
          'gen',
          'layers_len_name',
          'num_seed_name',
          'type_seed_name',
      ))
    )
    try:
      g_gen=_MII._load_from_format(gen['g'])
      l_gen=_MILI._load_from_format(gen['l'])
      t_gen=_MILI._load_from_format(gen['t'])
    except Exception:
      cls._load_err_raiser()
    return cls.__from_gen(
      (g_gen, l_gen, t_gen),
      layers_len_name=layers_len_name,
      num_seed_name=num_seed_name,
      type_seed_name=type_seed_name,
    )
