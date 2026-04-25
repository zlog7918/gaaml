import typing as t
import keras as krs
from . import consts as const
from .classes.NetIndividual import NetIndividual

__keras_activation_types: dict[int, t.Callable]={
  0: krs.activations.relu,
  1: krs.activations.tanh,
  2: krs.activations.sigmoid,
  3: krs.activations.leaky_relu,
}

__keras_optimalization_types: dict[int, type[krs.optimizers.Optimizer]]={
  0: krs.optimizers.SGD,
  1: krs.optimizers.Adam,
}

# __keras_losses_types: dict[int, type[krs.losses.Loss]]={
#   0: krs.losses.MeanSquaredError,
#   # 1: krs.losses.,
# }

def __cr_net_from_ind(net_ind: NetIndividual, input_size: int, output_size: int) -> krs.models.Model:
  # TODO
  params, layer_sizes, layer_types=net_ind.gen
  params, layer_sizes, layer_types=params.fenotype, layer_sizes.fenotype, layer_types.fenotype

  seq=krs.models.Sequential()
  seq.add(krs.layers.Input(shape=(input_size,)))
  for n, t in zip((*layer_sizes, output_size), layer_types):
    seq.add(krs.layers.Dense(n, activation=__keras_activation_types[t]))
  learning_rate=params[const.BIN_PART_LEARNING_RATE_NAME]/(2<<13)
  seq.compile(
    __keras_optimalization_types[params[const.BIN_PART_OPTIMIZER_NAME]](
      learning_rate=learning_rate
    ),
    krs.losses.MeanSquaredError(),
  )
  return seq

# def swith_backend(backend: t.Literal['torch']|t.Literal['tensorflow']) -> None: ...
#   # TODO:
#   # Is possible to switch between tensorflow and torch using:
#   # krs.config.set_backend()
def cr_net_from_ind(net_ind: NetIndividual, input_size: int, output_size: int) -> tuple[krs.models.Model, int, int]:
# def cr_net_from_ind(
#   net_ind: NetIndividual,
#   input_size: int,
#   output_size: int,
#   backend: t.Literal['torch']|t.Literal['tensorflow']='tensorflow',
# ) -> tuple[krs.models.Model, int, int]:
#   if krs.config.backend()!=backend:
#     swith_backend(backend)
  params, _, _=net_ind.gen
  params=params.fenotype
  model=__cr_net_from_ind(net_ind, input_size, output_size)
  return (model, params[const.BIN_PART_BATCH_NAME], params[const.BIN_PART_EPOCHS_NAME])
