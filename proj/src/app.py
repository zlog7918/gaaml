import math
import numpy as np
import typing as t
import random as rnd
from . import utils as util
from . import consts as const
from .classes.Population import Population
from .classes.MaxIntsListIndividual import MaxIntsListIndividual
# import keras as krs
# import tensorflow

IndType=MaxIntsListIndividual
RetType=MaxIntsListIndividual

@t.overload
def cr_network(
  training_data: np.ndarray,
  test_data: np.ndarray,
  /,*,
  population_size: int=...,
  number_of_generations: int=...,
  cross_rate: float=...,
  mutation_rate: float=...,
  plot: bool=False,
) -> RetType: ...
@t.overload
def cr_network(
  training_data: np.ndarray,
  validation_data: np.ndarray,
  test_data: np.ndarray,
  /,*,
  population_size: int=...,
  number_of_generations: int=...,
  cross_rate: float=...,
  mutation_rate: float=...,
  plot: bool=False,
) -> RetType: ...
def cr_network(
  training_data: np.ndarray,
  _validation_data: np.ndarray,
  _test_data: np.ndarray|None=None,
  /,*,
  population_size: int=const.POP_SIZE,
  number_of_generations: int=const.NUM_OF_GENERATIONS,
  cross_rate: float=const.CROSS_RATE,
  mutation_rate: float=const.MUTATE_RATE,
  plot: bool=False,
) -> RetType:
  test_data, validation_data=(
    (_validation_data, None)
      if _test_data is None else
    (_test_data, _validation_data)
  )
  del _validation_data, _test_data

  def f(x: list[int]) -> float:
    len_part=63-9*abs(len(x)-3)
    x_part=sum(4-abs(el-6) for el in x)
    return len_part+x_part
  def fitness(net_ind: IndType) -> float:
    # TODO: write fitness func
    feno=net_ind.fenotype
    ret=f(feno)+10
    return ret if ret>0 else 0

  pop=Population(
    population_size,
    lambda: IndType(
      rnd.randint(*const.MIN_MAX_LEN),
      (const.MIN_MAX_LEN, const.NEURON_TYPE),
    ),
    IndType.crossover,
    IndType.get_cp,
    fitness,
    cross_rate,
    mutation_rate,
  )

  maxs: list[float]=[0]*(const.NUM_OF_GENERATIONS+1)
  avgs: list[float]=[0]*(const.NUM_OF_GENERATIONS+1)
  mins: list[float]=[0]*(const.NUM_OF_GENERATIONS+1)

  max_sol: IndType|None=None
  min_sol: IndType|None=None
  max_of_max=-1
  min_of_min=float('inf')
  for i in range(number_of_generations+1):
    _max_sol, _min_sol, _max, _avg, _min=util.get_max_avg_min(pop)
    if _max>max_of_max:
      max_of_max=_max
      max_sol=_max_sol
    if _min<min_of_min:
      min_of_min=_min
      min_sol=_min_sol
    maxs[i]=_max
    avgs[i]=_avg
    mins[i]=_min
    if number_of_generations!=i:
      pop.next_generation()
  if max_sol is None or min_sol is None:
    raise Exception('number_of_generations: is too small')

  if plot:
    import matplotlib.pyplot as plt
    for i, (title, l) in enumerate(zip(('Max values', 'Avg values', 'Min values'), (maxs, avgs, mins))):
      i+=1
      plt.figure(i)
      plt.plot(list(range(len(l))), l, marker='o', color='b', linestyle='-')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.xlim((0, len(l)+1))
      plt.ylim((0, max_of_max))
      plt.title(title)
      plt.show()

  return max_sol

  # seq=krs.models.Sequential()
  # seq.compile(
  #   krs.optimizers.Adam(learning_rate=0.001), # type: ignore[arg-type]
  #   loss=None
  # )
  # seq.fit(
  #   epochs=1
  # )
