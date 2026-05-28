import gc
import typing as t
import random as rnd
from pathlib import Path
from tqdm.auto import tqdm
from . import _utils as util
from . import _consts as const
from .DirManager import DirManager
from concurrent.futures import ThreadPoolExecutor
from .MaxAvgMinHolder import MaxAvgMinHolder as MAMHolder

class Population(t.Generic[util.IndividualType]):
  __calc_avg_from_fittnesses: t.Callable[[list[float]], float]=staticmethod(lambda i: sum(i)/len(i))
  @property
  def population(self) -> list[util.IndividualType]:
    return self.__population[:]
  __fitnesses: MAMHolder[list[float]]
  @property
  def fitnesses_all(self) -> list[list[float]]:
    return self.__fitnesses.arr
  @property
  def fitnesses(self) -> list[float]:
    return self.__fitnesses.arr_v
  # __calc_fitnesses: t.Callable[[int], None]
  def __init__(
    self,
    pop_num: int,
    individual_factory: t.Callable[[], util.IndividualType],
    calc_fitness_func: t.Callable[[util.IndividualType, Path], float],
    cross_rate: float,
    mutate_rate: float,
    *,
    fitnesses_progress_output: tqdm|None,
    num_of_fit_calc: int=const.NUM_OF_FIT_CALC,
    max_worker_num: int=const.MAX_WORKERS,
    save_dir_path: Path|str|None=None,
  ) -> None:
    super().__init__()
    self.__population: list[util.IndividualType]=[individual_factory() for _ in range(pop_num)]
    def calc_fitness(ind: util.IndividualType, dir: Path) -> float:
      dir.mkdir(parents=True)
      fit=calc_fitness_func(ind, dir)
      return fit
    self.__fpo: tqdm|None=fitnesses_progress_output
    self.__calc_fitness: t.Callable[
      [util.IndividualType, Path],
      float,
    ]=calc_fitness
    self.__cross_rate: float=cross_rate
    self.__mutate_rate: float=mutate_rate
    self.__max_worker_num: int=max_worker_num
    self.__num_of_fit_calc: int=num_of_fit_calc
    self.__yield_fits=(
      self.__yield_fits_multi
        if self.__max_worker_num>1 else
      self.__yield_fits_seq
    )
    self.__dir=DirManager(save_dir_path)
    self.__calc_fitnesses(0)

  def set_dir(self, dir: Path|str) -> None:
    self.__dir.path=dir

  def __calc_ind_fitnesses(
    self,
    gen_num: int,
    ind_i: int,
    ind: util.IndividualType,
  ) -> list[float]:
    path = self.__dir.path/f'gen_{gen_num}'/f'ind_{ind_i}'
    ind.save_to(path/'model.gen')
    return [
      self.__calc_fitness(ind, path/f'iter_{i}')
        for i in
      range(self.__num_of_fit_calc)
    ]

  def __yield_fits_multi(self, gen_num: int):
    with ThreadPoolExecutor(max_workers=self.__max_worker_num) as executor:
      future_fits=[
        executor.submit(
          self.__calc_ind_fitnesses,
          gen_num,
          ind_i,
          ind,
        )
          for ind_i, ind in
        enumerate(self.__population)
      ]
      for future in future_fits:
        yield future.result()

  def __yield_fits_seq(self, gen_num: int):
    for ind_i, ind in enumerate(self.__population):
      yield self.__calc_ind_fitnesses(
        gen_num,
        ind_i,
        ind,
      )

  def __calc_fitnesses(self, gen_num: int) -> None:
    fitnesses=MAMHolder[list[float]](len(self.__population), self.__calc_avg_from_fittnesses)
    for fit in self.__yield_fits(gen_num):
      if self.__fpo is not None:
        self.__fpo.update()
      fitnesses.append(fit)
    self.__fitnesses=fitnesses
    gc.collect()

  @staticmethod
  def _calc_to_add(fitnesses: MAMHolder[list[float]]) -> float:
    # add more elaborate way, so if fitnesses for example are: [0, 0, 0, 5, 0], id does not always cross one and the same individual
    # a=\frac{
    #   2S
    # }{
    #   n(k+7)-k(k+9)
    # }
    to_add=float('inf')
    for v in fitnesses.arr_v:
      if v>0 and v<to_add:
        to_add=v
    to_add*=.01
    return to_add

  def _selection(self) -> tuple[util.IndividualType, util.IndividualType]:
    if self.__fitnesses.sum==0:
      return rnd.choice(self.__population), rnd.choice(self.__population)
      # fitness_values=[fit+1 for fit in fitness_values]
    to_add=self._calc_to_add(self.__fitnesses) if self.__fitnesses.zero_count>0 else 0

    probs=[]
    _sum=0
    for p in ((fit+to_add)/self.__fitnesses.sum for fit in self.__fitnesses.arr_v):
      _sum+=p
      probs.append(_sum)
    del _sum

    i1=util.get_i_in_range(probs, rnd.random())
    i2=util.get_i_in_range(probs, rnd.random())
    return self.__population[i1], self.__population[i2]

  @staticmethod
  def mutate(individual: util.IndividualType, mutation_rate: float) -> util.IndividualType:
    if rnd.random()<mutation_rate:
      individual.mutate()
    return individual

  @staticmethod
  def crossover(parent1: util.IndividualType, parent2: util.IndividualType, cross_rate: float) -> tuple[util.IndividualType, util.IndividualType]:
    ind_type=type(parent1)
    if rnd.random()<cross_rate:
      cp=ind_type.get_cp(parent1, parent2)
      return ind_type.crossover(parent1, parent2, cp)
    return parent1, parent2

  def next_generation(self, gen_num: int) -> None:
    if self.__fpo is not None:
      self.__fpo.reset()
    new_generation=[]
    pop_len=len(self.__population)
    for _ in range(int((pop_len+1)/2)):
      parent1, parent2=self._selection()
      for child in self.crossover(parent1, parent2, self.__cross_rate):
        child=self.mutate(child, self.__mutate_rate)
        new_generation.append(child)

    self.__population=new_generation[:pop_len]
    self.__calc_fitnesses(gen_num)

  def get_max_avg_min(self) -> tuple[util.IndividualType, util.IndividualType, float, float, float]:
    fitnesses=self.__fitnesses
    max_avg_min=(
      fitnesses.max_v,
      fitnesses.avg,
      fitnesses.min_v,
    )

    max_sol: util.IndividualType=self.__population[fitnesses.max_i]
    min_sol: util.IndividualType=self.__population[fitnesses.min_i]
    return max_sol, min_sol, *max_avg_min
