import numpy as np
import typing as t
from . import consts as const
from .classes.Population import Population
from .classes.Generations import Generations
from .classes.NetIndividual import NetIndividual
# import keras as krs
# import tensorflow

IndType=NetIndividual
RetType=NetIndividual

def fitness(_f: t.Callable[[IndType], float], net_ind: IndType) -> float:
  ret=_f(net_ind)
  return ret if ret>0 else 0

@t.overload
def cr_network(
  training_data: np.ndarray,
  test_data: np.ndarray,
  /,*,
  population_size: int=...,
  number_of_generations: int=...,
  cross_rate: float=...,
  mutation_rate: float=...,
  fitness_func: t.Callable[[
    t.Callable[[IndType], float],
    IndType
  ], float]=...,
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
  fitness_func: t.Callable[[
    t.Callable[[IndType], float],
    IndType
  ], float]=...,
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
  fitness_func: t.Callable[[
    t.Callable[[IndType], float],
    IndType
  ], float]=fitness,
  plot: bool=False,
) -> RetType:
  test_data, validation_data=(
    (_validation_data, None)
      if _test_data is None else
    (_test_data, _validation_data)
  )
  del _validation_data, _test_data

  def _f(training_data: np.ndarray, validation_data: np.ndarray|None, test_data: np.ndarray) -> t.Callable[[IndType], float]:
    def f(net_ind: IndType) -> float:
      params, layer_sizes, layer_types=net_ind.gen
      params, layer_sizes, layer_types=params.fenotype, layer_sizes.fenotype, layer_types.fenotype
      return 0
    return f
  f=_f(training_data, validation_data, test_data)
  pop=Population(
    population_size,
    lambda: IndType(
      const.BIN_PART_LIST_LEN,
      const.BIN_PART_NEURON_NUM_SEED,
      const.BIN_PART_NEURON_TYPE_SEED,
      (
        const.BIN_PART_REST,
        const.NEURON_NUM,
        const.NEURON_TYPE,
      ),
    ),
    IndType.crossover,
    IndType.get_cp,
    lambda x: fitness_func(f, x),
    cross_rate,
    mutation_rate,
  )

  generations=Generations(pop, number_of_generations)
  (
    (max_sol, min_sol),
    (max_of_max, min_of_min),
    (maxs, avgs, mins),
  )=generations.go_through_generations()

  if plot:
    if max_of_max!=0:
      print('Maksymalna wartość:', max_of_max)
    import matplotlib.pyplot as plt
    for i, (title, l) in enumerate(zip(
      ('Max values', 'Avg values', 'Min values'),
      (maxs, avgs, mins)
    )):
      i+=1
      plt.figure(i)
      plt.plot(list(range(len(l))), l, marker='o', color='b', linestyle='-')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.xlim((0, len(l)+1))
      if max_of_max==0:
        plt.ylim((0, .5))
      else:
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
