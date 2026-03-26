import typing as t
from . import _utils as util
from .Population import Population

class Generations(t.Generic[util.IndividualType]):
  __max_sol: util.IndividualType|None=None
  __min_sol: util.IndividualType|None=None
  __max_of_max=-1.
  __min_of_min=float('inf')
  __curr_generations=0
  def __init__(self, pop: Population[util.IndividualType], max_num_gen: int) -> None:
    super().__init__()
    # assert num_gen>0
    if max_num_gen<1:
      raise ValueError('max_num_gen: is too small, it should at least equal 1')
    self.__pop=pop
    self.__max_num_gen=max_num_gen
    self.__maxs: list[float]=[.0]*(self.__max_num_gen+1)
    self.__avgs: list[float]=[.0]*(self.__max_num_gen+1)
    self.__mins: list[float]=[.0]*(self.__max_num_gen+1)
    self.__arrange_min_max(self.__curr_generations)

  def go_through_generations(self, num_gen: int|None=None) -> None:
    if num_gen is None:
      num_gen=self.__max_num_gen-self.__curr_generations
    elif num_gen>self.__max_num_gen-self.__curr_generations:
      num_gen=self.__max_num_gen-self.__curr_generations
    if num_gen<1:
      raise IndexError('Tried to add next generation(s) after reaching max number of them')
    for _ in range(num_gen):
      self.__pop.next_generation()
      self.__curr_generations+=1
      self.__arrange_min_max(self.__curr_generations)

  def __arrange_min_max(self, i: int) -> None:
    _max_sol, _min_sol, _max, _avg, _min=self.__pop.get_max_avg_min()
    if _max>self.__max_of_max:
      self.__max_of_max=_max
      self.__max_sol=_max_sol
    if _min<self.__min_of_min:
      self.__min_of_min=_min
      self.__min_sol=_min_sol
    self.__maxs[i]=_max
    self.__avgs[i]=_avg
    self.__mins[i]=_min

  @property
  def curr_generations(self) -> int:
    return self.__curr_generations

  def get_statistics(self) -> tuple[tuple[util.IndividualType, util.IndividualType], tuple[float, float], tuple[list[float], list[float], list[float]]]:
    assert self.__max_sol is not None
    assert self.__min_sol is not None
    return (
      (self.__max_sol, self.__min_sol),
      (self.__max_of_max, self.__min_of_min),
      (self.__maxs[:self.__curr_generations+1], self.__avgs[:self.__curr_generations+1], self.__mins[:self.__curr_generations+1]),
    )
