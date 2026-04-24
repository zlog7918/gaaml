import json
import warnings
import numpy as np
import typing as t
import keras as krs
from pathlib import Path
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

count=0

def __f(training_data: np.ndarray, validation_data: np.ndarray|None, test_data: np.ndarray, number_of_attributes: int) -> t.Callable[[IndType, Path], float]:
  if training_data.shape[1]!=test_data.shape[1]:
    raise ValueError('training_data and test_data do not have the same number of attributes in data or output')
  if validation_data is not None and training_data.shape[1]!=validation_data.shape[1]:
    raise ValueError('validation_data does not have the same number of attributes in data or output as training_data and test_data')
  training_data_x, training_data_y=training_data[:,:number_of_attributes], training_data[:,number_of_attributes:]
  _validation_data=None if validation_data is None else (validation_data[:,:number_of_attributes], validation_data[:,number_of_attributes:])
  del validation_data
  test_data_x, test_data_y=test_data[:,:number_of_attributes], test_data[:,number_of_attributes:]

  def f(
    net_ind: IndType,
    dir: Path,
    verbose: t.Literal[0]|t.Literal[1]|t.Literal[2]|t.Literal['auto']='auto',
  ) -> float:
    global count
    count+=1
    model, batch_size, epochs=util.cr_net_from_ind(net_ind, training_data_x.shape[1], training_data_y.shape[1])
    fit_ret=model.fit(
      training_data_x,
      training_data_y,
      batch_size=batch_size,
      epochs=epochs,
      validation_data=_validation_data,
      verbose=verbose, # type: ignore
    ) # throws 3 warnings: DeprecationWarning: __array__ implementation doesn't accept a copy keyword, so passing copy=False failed.

    with open(dir/'model_meta.data', 'x') as meta:
      json.dump(
        {
          'id': count,
          'epoch': epochs,
          'batch': batch_size,
          'backend': krs.config.backend(),
          'hidden_len': len(model.get_weights())//2-1,
          'fit_ret': str(fit_ret),
        },
        meta,
      )
    with warnings.catch_warnings(category=DeprecationWarning):
      warnings.filterwarnings(
        'ignore',
        message='__array__ implementation doesn\'t accept a copy keyword',
        category=DeprecationWarning,
      )
      model.save_weights(dir/'model.weights.h5')

    ret=model.evaluate(
      test_data_x,
      test_data_y,
      verbose=verbose, # type: ignore
    )
    return ret
  return f

def fitness(_f: t.Callable[[IndType, Path], float], net_ind: IndType, dir: Path) -> float:
  ret=_f(net_ind, dir)
  ret=1/(ret+1)
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
  fitness_func: t.Callable[[
    t.Callable[[IndType, Path], float],
    IndType,
    Path,
  ], float]=...,
  plot: bool=False,
  max_worker_num: int=...,
  num_of_fittnesses_calc: int=...,
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
  fitness_func: t.Callable[[
    t.Callable[[IndType, Path], float],
    IndType,
    Path,
  ], float]=...,
  plot: bool=False,
  max_worker_num: int=...,
  num_of_fittnesses_calc: int=...,
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
  fitness_func: t.Callable[[
    t.Callable[[IndType, Path], float],
    IndType,
    Path,
  ], float]=fitness,
  plot: bool=False,
  max_worker_num: int=Pop_const.MAX_WORKERS,
  num_of_fittnesses_calc: int=Pop_const.NUM_OF_FIT_CALC,
) -> RetType:
  test_data, validation_data=(
    (_validation_data, None)
      if _test_data is None else
    (_test_data, _validation_data)
  )
  del _validation_data, _test_data
  f=__f(training_data, validation_data, test_data, number_of_attributes)
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
    lambda x, dir: fitness_func(f, x, dir),
    cross_rate,
    mutation_rate,
    save_dir_path=save_dir_path,
    max_worker_num=max_worker_num,
    num_of_fit_calc=num_of_fittnesses_calc,
  )

  generations=Generations(pop, number_of_generations)
  generations.go_through_generations()
  (
    (max_sol, min_sol),
    (max_of_max, min_of_min),
    (maxs, avgs, mins),
  )=generations.get_statistics()

  if plot:
    ylim=.5
    if max_of_max!=0:
      ylim=max_of_max
      # print('Maksymalna wartość:', max_of_max)
    import matplotlib.pyplot as plt
    for i, (title, l) in enumerate(zip(
      (f'Max values {'no max' if max_of_max==0 else f'(max: {max_of_max})'}', 'Avg values', 'Min values'),
      (maxs, avgs, mins)
    )):
      i+=1
      plt.figure(i)
      plt.plot(list(range(len(l))), l, marker='o', color='b', linestyle='-')
      plt.xlabel('x')
      plt.ylabel('y')
      plt.xlim((0, len(l)+1))
      plt.ylim((0, ylim))
      plt.title(title)
      plt.show()

  return generations
