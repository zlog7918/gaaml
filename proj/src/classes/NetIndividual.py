import typing as t
import random as rnd
from .Individual import Individual
from .MaxIntsIndividual import MaxIntsIndividual
from .MaxIntsListIndividual import MaxIntsListIndividual

class NetIndividual(Individual[tuple[MaxIntsIndividual, MaxIntsListIndividual, MaxIntsListIndividual]]):
  GenShemaType=tuple[MaxIntsIndividual.GenShemaType, MaxIntsListIndividual.GenShemaType, MaxIntsListIndividual.GenShemaType]
  CPType=tuple[MaxIntsIndividual.CPType, MaxIntsListIndividual.CPType, MaxIntsListIndividual.CPType]
  @t.overload
  def __init__(
    self,
    layers_len: tuple[str, tuple[int, int, int]],
    gen_shema: GenShemaType,
    /,
  ): ...
  @t.overload
  def __init__(
    self,
    a: "NetIndividual",
    b: "NetIndividual",
    /, *,
    cross_point: CPType
  ): ...
  def __init__(
    self,
    a: "tuple[str, tuple[int, int, int]]|NetIndividual",
    b: "GenShemaType|NetIndividual",
    /, *,
    cross_point: CPType|None=None,
  ):
    if isinstance(a, tuple) and isinstance(b, tuple):
      layers_len_name, _=a
      g_shema, l_shema, t_shema=b
      names={name for name,_ in g_shema}
      if layers_len_name in names:
        raise Exception('Item names must be unique')
      if len(names)!=len(g_shema):
        raise Exception('Item names must be unique')
      g_shema=(a, *g_shema)
      g=MaxIntsIndividual(g_shema)
      list_len=g.fenotype[layers_len_name]
      l=MaxIntsListIndividual(list_len, l_shema)
      t=MaxIntsListIndividual(list_len, t_shema)
      super().__init__((g, l, t))
      return
    if isinstance(a, tuple) or isinstance(b, tuple) or cross_point is None:
      raise Exception('Illegal argument options')
    (
      (ag, al, at),
      (bg, bl, bt),
      (cg, cl, ct),
    )=a.gen, b.gen, cross_point
    g=MaxIntsIndividual(ag, bg, cross_point=cg)
    l=MaxIntsListIndividual(al, bl, cross_point=cl)
    t=MaxIntsListIndividual(at, bt, cross_point=ct)
    super().__init__((g, l, t))

  def mutate(self) -> None:
    n=rnd.randint(1, len(self.gen))
    for i in rnd.sample(range(len(self.gen)), n):
      self.gen[i].mutate()

  _NI=t.TypeVar('_NI', bound="NetIndividual")
  @classmethod
  def get_cp(cls: type[_NI], a: _NI, b: _NI) -> CPType:
    ag, al, at=a.gen
    bg, bl, bt=b.gen
    return (
      MaxIntsIndividual.get_cp(ag, bg),
      MaxIntsListIndividual.get_cp(al, bl),
      MaxIntsListIndividual.get_cp(at, bt),
    )

  @classmethod
  def crossover(cls: type[_NI], a: _NI, b: _NI, cp: CPType) -> tuple[_NI, _NI]:
    c0, (ac1, bc1), (ac2, bc2)=cp
    return (
      cls(a, b, cross_point=cp),
      cls(b, a, cross_point=(c0, (bc1, ac1), (bc2, ac2))),
    )
