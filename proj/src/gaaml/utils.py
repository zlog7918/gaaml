import typing as t
import keras as krs
# import torch.nn as nn
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

# __torch_activation_types: dict[int, type[nn.Module]]={
#   0: nn.ReLU,
#   1: nn.Tanh,
#   2: nn.Sigmoid,
#   3: nn.Softmax,
# }

def __cr_net_from_ind_keras(net_ind: NetIndividual, input_size: int, output_size: int) -> krs.models.Model:
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

# def __cr_net_from_ind_torch(net_ind: NetIndividual, input_size: int, output_size: int) -> nn.Sequential:
#   # TODO
#   params, layer_sizes, layer_types=net_ind.gen
#   params, layer_sizes, layer_types=params.fenotype, layer_sizes.fenotype, layer_types.fenotype
#   modules=[]
#   for pn, n, t in zip((input_size, *layer_sizes), (*layer_sizes, output_size), layer_types):
#     modules.append(nn.Linear(pn, n))
#     modules.append(__torch_activation_types[t]())
#   seq=nn.Sequential(*modules)
#   return seq

# @t.overload
# def cr_net_from_ind(net_ind: NetIndividual, input_size: int, output_size: int, type: t.Literal['torch']) -> tuple[nn.Sequential, int, int]: ...
# @t.overload
# def cr_net_from_ind(net_ind: NetIndividual, input_size: int, output_size: int, type: t.Literal['keras']) -> tuple[krs.models.Model, int, int]: ...
def cr_net_from_ind(net_ind: NetIndividual, input_size: int, output_size: int) -> tuple[krs.models.Model, int, int]:
# def cr_net_from_ind(net_ind: NetIndividual, input_size: int, output_size: int, type: str='keras') -> tuple[krs.models.Model|nn.Sequential, int, int]:
#   func=__cr_net_from_ind_keras if type=='keras' else __cr_net_from_ind_torch
  func=__cr_net_from_ind_keras
  params, _, _=net_ind.gen
  params=params.fenotype
  model=func(net_ind, input_size, output_size)
  return (model, params[const.BIN_PART_BATCH_NAME], params[const.BIN_PART_EPOCHS_NAME])
