import numpy as np
import typing as t
from . import consts as const
# import keras as krs
# import tensorflow

@t.overload
def cr_network(
  training_data: np.ndarray,
  test_data: np.ndarray,
  /,*,
  population_size: int=...,
  number_of_generations: int=...,
  cross_rate: float=...,
  mutation_rate: float=...,
) -> None: ...
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
) -> None: ...
def cr_network(
  training_data: np.ndarray,
  validation_data: np.ndarray,
  test_data: np.ndarray|None=None,
  /,*,
  population_size: int=const.POP_SIZE,
  number_of_generations: int=const.NUM_OF_GENERATIONS,
  cross_rate: float=const.CROSS_RATE,
  mutation_rate: float=const.MUTATE_RATE,
) -> None:
  # seq=krs.models.Sequential()
  # seq.compile(
  #   krs.optimizers.Adam(learning_rate=0.001), # type: ignore[arg-type]
  #   loss=None
  # )
  # seq.fit(
  #   epochs=1
  # )

  pass
