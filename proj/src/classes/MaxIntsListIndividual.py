import typing as t
from . import _utils as util
from .Individual import Individual
from .ListIndividual import ListIndividual

class MaxIntsListIndividual(Individual[ListIndividual]):
  GenSchemaType=tuple[tuple[int, int], tuple[int, int, int]]
  CPType=ListIndividual.CPType
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
      a.__same_or_err(b)
      super().__init__(ListIndividual(a.gen, b.gen, cross_point=cross_point))
      self.schema=a.schema
    self._update_fenotype()

  def mutate(self) -> None:
    self.gen.mutate()
    self._update_fenotype()

  def _update_fenotype(self) -> None:
    self.fenotype: list[int]=[]
    min_v, max_v=self.schema
    for idx in range(0, len(self.gen.gen), self.gen.item_size):
      self.fenotype.append(
        util.correct_gen_to_min_max(
          self.gen.gen[idx:idx+self.gen.item_size],
          min_v,
          max_v,
        )
      )

  _MILI=t.TypeVar('_MILI', bound="MaxIntsListIndividual")
  def __same_or_err(self: _MILI, o: _MILI) -> None:
    if self.schema!=o.schema:
      raise Exception('First and second solution do not have equal configuration')

  @classmethod
  def get_cp(cls: type[_MILI], a: _MILI, b: _MILI) -> CPType:
    a.__same_or_err(b)
    return ListIndividual.get_cp(a.gen, b.gen)

  @classmethod
  def crossover(cls: type[_MILI], a: _MILI, b: _MILI, cp: CPType) -> tuple[_MILI, _MILI]:
    return (cls(a, b, cross_point=cp), cls(b, a, cross_point=(cp[1], cp[0])))
