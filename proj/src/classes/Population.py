import typing as t
import random as rand
from . import _utils as util

class Population(t.Generic[util.IndividualType, util.CpType]):
  population: list[util.IndividualType]
  _get_cp: t.Callable[[util.IndividualType, util.IndividualType], util.CpType]
  _crossover: t.Callable[[util.IndividualType, util.IndividualType, util.CpType], tuple[util.IndividualType, util.IndividualType]]
  fitness: t.Callable[[util.IndividualType], float]
  fitnesses: list[float]
  cross_rate: float
  mutate_rate: float
  def __init__(
    self,
    pop_num: int,
    individual_gen: t.Callable[[], util.IndividualType],
    crossover: t.Callable[[util.IndividualType, util.IndividualType, util.CpType], tuple[util.IndividualType, util.IndividualType]],
    crossover_point: t.Callable[[util.IndividualType, util.IndividualType], util.CpType],
    fitness: t.Callable[[util.IndividualType], float],
    cross_rate: float,
    mutate_rate: float,
  ):
    super().__init__()
    self.population=[individual_gen() for _ in range(pop_num)]
    self._crossover=crossover
    self._get_cp=crossover_point
    self.fitness=fitness
    self.cross_rate=cross_rate
    self.mutate_rate=mutate_rate
    self._calc_fitnesses()

  def _calc_fitnesses(self) -> None:
    # TODO: write multi-thread version of calc fitness func
    self.fitnesses=[self.fitness(ind) for ind in self.population]
    self.fitnesses_sum=sum(self.fitnesses)

  def _proportional_selection(self) -> tuple[util.IndividualType, util.IndividualType]:
    if self.fitnesses_sum==0:
      return rand.choice(self.population), rand.choice(self.population)
      # fitness_values=[fit+1 for fit in fitness_values]

    prob=[fit/self.fitnesses_sum for fit in self.fitnesses]
    probs=[]
    _sum=0
    for p in prob:
      _sum+=p
      probs.append(_sum)
    del prob, _sum

    i1=util.get_i_in_range(probs, rand.random())
    i2=util.get_i_in_range(probs, rand.random())
    return self.population[i1], self.population[i2]

  def crossover(self, parent1: util.IndividualType, parent2: util.IndividualType) -> tuple[util.IndividualType, util.IndividualType]:
    if rand.random()<self.cross_rate:
      cp=self._get_cp(parent1, parent2)
      return self._crossover(parent1, parent2, cp)
    return parent1, parent2

  def next_generation(self) -> None:
    new_generation=[]
    pop_len=len(self.population)
    for _ in range(int((pop_len+1)/2)):
      parent1, parent2=self._proportional_selection()
      for child in self.crossover(parent1, parent2):
        if rand.random()<self.mutate_rate:
          child.mutate()
        new_generation.append(child)

    self.population=new_generation[:pop_len]
    self._calc_fitnesses()
