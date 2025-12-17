import typing as t
from .classes import _utils as util
from .classes.Population import Population

def get_max_avg_min(pop: Population[util.IndividualType, util.CpType]) -> tuple[t.Optional[util.IndividualType], t.Optional[util.IndividualType], float, float, float]:
  n=len(pop.population)
  _min=float('inf')
  _max=-1
  _sum=0

  max_sol: t.Optional[util.IndividualType]=None
  min_sol: t.Optional[util.IndividualType]=None
  for fit, sol in zip(pop.fitnesses, pop.population):
    if fit>_max:
      max_sol=sol
      _max=fit
    if fit<_min:
      min_sol=sol
      _min=fit
    _sum+=fit
  return max_sol, min_sol, _max, _sum/n, _min