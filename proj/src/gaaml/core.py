import numpy as np
import typing as t
from pathlib import Path
from tqdm.auto import tqdm
from . import utils as util
from . import consts as const
from .classes.Population import (
  const as Pop_const,
  Population,
)
from .classes.Generations import Generations
from .classes.NetIndividual import NetIndividual

IndType=NetIndividual
RetType=Generations

def fitness_corrector(net_ind: IndType, fit: float) -> float:
  ret=1/(fit+1)
  ret=ret if ret>0 else 0
  return ret

@t.overload
def cr_network(
  training_data: np.ndarray,
  test_data: np.ndarray,
  /,*,
  save_dir_path: Path|str,
  number_of_attributes: int=...,
  population_size: int=...,
  number_of_generations: int=...,
  cross_rate: float=...,
  mutation_rate: float=...,
  fitness_func: t.Callable[
    [IndType, float],
    float,
  ]=fitness_corrector,
  max_worker_num: int=...,
  num_of_fittnesses_calc: int=...,
  output_progress: bool=True,
  plot: bool=False,
) -> RetType: ...
@t.overload
def cr_network(
  training_data: np.ndarray,
  validation_data: np.ndarray,
  test_data: np.ndarray,
  /,*,
  save_dir_path: Path|str,
  number_of_attributes: int=...,
  population_size: int=...,
  number_of_generations: int=...,
  cross_rate: float=...,
  mutation_rate: float=...,
  fitness_func: t.Callable[
    [IndType, float],
    float,
  ]=fitness_corrector,
  max_worker_num: int=...,
  num_of_fittnesses_calc: int=...,
  output_progress: bool=True,
  plot: bool=False,
) -> RetType: ...
def cr_network(
  training_data: np.ndarray,
  _validation_data: np.ndarray,
  _test_data: np.ndarray|None=None,
  /,*,
  save_dir_path: Path|str,
  number_of_attributes: int=-1,
  population_size: int=const.POP_SIZE,
  number_of_generations: int=const.NUM_OF_GENERATIONS,
  cross_rate: float=const.CROSS_RATE,
  mutation_rate: float=const.MUTATE_RATE,
  fitness_func: t.Callable[
    [IndType, float],
    float,
  ]=fitness_corrector,
  max_worker_num: int=Pop_const.MAX_WORKERS,
  num_of_fittnesses_calc: int=Pop_const.NUM_OF_FIT_CALC,
  output_progress: bool=True,
  plot: bool=False,
) -> RetType:
  training_data=np.asarray(training_data)
  validation_data, test_data=(
    (None, np.asarray(_validation_data))
      if _test_data is None else
    (np.asarray(_validation_data), np.asarray(_test_data))
  )
  del _validation_data, _test_data
  fit_func=util.get_fit_func(training_data, validation_data, test_data, number_of_attributes)
  if output_progress is True:
    bar1=tqdm(total=number_of_generations, desc='Generations', position=0, mininterval=0)
    bar2=tqdm(total=population_size, desc='Calculated fitnesses', position=1, mininterval=0)
  else:
    bar1=bar2=None

  pop_args, pop_kwargs=(
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
    lambda net_ind, dir: t.cast(
      tuple[float, float],
      tuple(
        fitness_func(net_ind, r)
          for r in
        fit_func(net_ind, dir)
      )
    ),
    cross_rate,
    mutation_rate
  ), {
    'fitnesses_progress_output': bar2,
    'save_dir_path': save_dir_path,
    'max_worker_num': max_worker_num,
    'num_of_fit_calc': num_of_fittnesses_calc,
  }

  generations=Generations(number_of_generations, bar1, Population, *pop_args, **pop_kwargs)
  generations.go_through_generations()
  if bar1 is not None and bar2 is not None:
    bar2.close()
    bar1.close()

  if plot:
    (
      (_, _),
      (max_of_max, _),
      (maxs, avgs, mins),
    )=generations.get_statistics()
    ylim=.5
    if max_of_max!=0:
      ylim=max_of_max
      # print('Maksymalna wartość:', max_of_max)
    import matplotlib.pyplot as plt
    for i, (title, l) in enumerate(zip(
      (f'Max values {"no max" if max_of_max==0 else f"(max: {max_of_max})"}', 'Avg values', 'Min values'),
      (maxs, avgs, mins)
    )):
      i+=1
      plt.figure(i)
      plt.plot(list(range(len(l))), l, marker='o', color='b', linestyle='-')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.xlim((0, len(l)))
      plt.ylim((0, ylim))
      plt.title(title)
      plt.show()

  return generations
