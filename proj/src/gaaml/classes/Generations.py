import typing as t
from . import _utils as util
from .Population import Population

class Generations(t.Generic[util.IndividualType]):
  max_sol: util.IndividualType|None=None
  min_sol: util.IndividualType|None=None
  max_of_max=-1.
  min_of_min=float('inf')
  def __init__(self, pop: Population[util.IndividualType], num_gen: int) -> None:
    super().__init__()
    # assert num_gen>0
    if num_gen<1:
      raise ValueError('num_gen: is too small, it should at least equal 1')
    self.pop=pop
    self.num_gen=num_gen
    self.maxs: list[float]=[0]*(self.num_gen+1)
    self.avgs: list[float]=[0]*(self.num_gen+1)
    self.mins: list[float]=[0]*(self.num_gen+1)

  def go_through_generations(self) -> tuple[tuple[util.IndividualType, util.IndividualType], tuple[float, float], tuple[list[float], list[float], list[float]]]:
    self.arrange_min_max(0)
    for i in range(1, self.num_gen+1):
      self.pop.next_generation()
      self.arrange_min_max(i)
    assert self.max_sol is not None
    assert self.min_sol is not None
    return (
      (self.max_sol, self.min_sol),
      (self.max_of_max, self.min_of_min),
      (self.maxs, self.avgs, self.mins),
    )

  def arrange_min_max(self, i: int) -> None:
    _max_sol, _min_sol, _max, _avg, _min=self.pop.get_max_avg_min()
    if _max>self.max_of_max:
      self.max_of_max=_max
      self.max_sol=_max_sol
    if _min<self.min_of_min:
      self.min_of_min=_min
      self.min_sol=_min_sol
    self.maxs[i]=_max
    self.avgs[i]=_avg
    self.mins[i]=_min
