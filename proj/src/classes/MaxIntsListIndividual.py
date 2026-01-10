import typing as t
from . import _utils as util
from .Individual import Individual
from .ListIndividual import ListIndividual

class MaxIntsListIndividual(Individual[ListIndividual]):
  GenSchemaType: t.TypeAlias=tuple[tuple[int, int], util.BitSize_Min_Max]
  CPType: t.TypeAlias=ListIndividual.CPType
  @t.overload
  def __init__(self, num_items: int, schema: GenSchemaType, /): ...
  @t.overload
  def __init__(self, a: "MaxIntsListIndividual", b: "MaxIntsListIndividual", /, *, cross_point: CPType): ...
  def __init__(self, a: "int|MaxIntsListIndividual", b: "GenSchemaType|MaxIntsListIndividual", /, *, cross_point: CPType|None=None):
    if isinstance(a, int) and isinstance(b, tuple):
      min_max_len, (elem_size, _max, _min)=b
      super().__init__(ListIndividual(a, (min_max_len, elem_size)))
      self.schema=_max, _min
    else:
      if isinstance(a, int) or isinstance(b, tuple) or cross_point is None:
        raise Exception('Illegal argument options')
      a._same_or_err(b)
      super().__init__(ListIndividual(a.gen, b.gen, cross_point=cross_point))
      self.schema=a.schema
    self._update_fenotype()

  def mutate(self) -> None:
    self.gen.mutate()
    self._update_fenotype()

  def _update_fenotype(self) -> None:
    self.fenotype: list[int]=self.get_fenotype(self.gen, *self.schema)

  _MILI=t.TypeVar('_MILI', bound="MaxIntsListIndividual")
  def _same_or_err(self: _MILI, o: _MILI) -> None:
    if self.schema!=o.schema:
      raise Exception('First and second solution do not have equal configuration')

  @staticmethod
  def get_fenotype(gen: ListIndividual, min_v: int, max_v: int) -> list[int]:
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
    a._same_or_err(b)
    return ListIndividual.get_cp(a.gen, b.gen)

  @classmethod
  def crossover(cls: type[_MILI], a: _MILI, b: _MILI, cp: CPType) -> tuple[_MILI, _MILI]:
    return (cls(a, b, cross_point=cp), cls(b, a, cross_point=(cp[1], cp[0])))
