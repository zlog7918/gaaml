import numpy as np
import typing as t
from . import utils as util
from . import consts as const
from .classes.Population import Population
from .classes.NetIndividual import NetIndividual
# import keras as krs
# import tensorflow

IndType=NetIndividual
RetType=NetIndividual

def fitness(_f: t.Callable[[IndType], float], net_ind: IndType) -> float:
  ret=1000-_f(net_ind)/8
  return ret if ret>0 else 0

@t.overload
def cr_network(
  training_data: np.ndarray,
  test_data: np.ndarray,
  /, *,
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
  /, *,
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
  /, *,
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

  def f(net_ind: IndType) -> float:
    g, l, t=net_ind.gen
    x, y=g.fenotype['x'], g.fenotype['y']
    length=g.fenotype[const.BIN_PART_LIST_LEN[0]]
    num_list, type_list=l.fenotype, t.fenotype

    _xy=(x-.6)**2+(y-.4)**2
    _len=abs(length-6)
    _num=sum(abs(n-20) for n in num_list)
    _type=sum(abs(t-4) for t in type_list)
    return _xy+_len+_type

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
