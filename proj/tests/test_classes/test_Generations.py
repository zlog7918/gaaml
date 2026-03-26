import pytest
import typing as t
import itertools
from gaaml.classes.Generations import Generations as G
from gaaml.classes.Population import Population as _P
from gaaml.classes.Individual import _BaseIndividual as _BI
# from gaaml.classes.GenIndividual import GenIndividual as _GI

class DummyInd(_BI["DummyInd", int, tuple[int, float], float]):
  def __init__(self, gen_int: int, gen_float: float) -> None:
    super().__init__((gen_int, gen_float), lambda gen: gen[1])
  @t.override
  def mutate(self) -> None: ...

  _DI=t.TypeVar('_DI', bound="DummyInd")
  @classmethod
  @t.override
  def get_cp(cls: type[_DI], a: _DI, b: _DI) -> int:
    return 0
  @classmethod
  @t.override
  def crossover(cls: type[_DI], a: _DI, b: _DI, cp: int) -> tuple[_DI, _DI]:
    (a_gen0, a_gen1)=a._gen
    (b_gen0, b_gen1)=b._gen
    return cls(a_gen0+1, a_gen1+.1), cls(b_gen0+1, b_gen1+.1)

class DummyPop(_P[DummyInd]):
  gen: int=0
  def __init__(self, pop_num: int, individual_factory: t.Callable[[], DummyInd]) -> None:
    calc_fitness_func: t.Callable[[DummyInd], float]=lambda ind: ind.gen
    super().__init__(pop_num, individual_factory, calc_fitness_func, 1., .0, max_worker_num=1)
  @t.override
  def next_generation(self) -> None:
    self.gen+=1
    self.__selection_list=iter(self.population)
    return super().next_generation()
  @t.override
  def _selection(self) -> tuple[DummyInd, DummyInd]:
    ind1=next(self.__selection_list)
    # if self.__selection_list.
    try:
      ind2=next(self.__selection_list)
    except StopIteration:
      ind2=self.population[0]
    return ind1, ind2

@pytest.mark.parametrize(
  ('pop_num', 'expected_max_avg_min'),
  [
    (9, ([4], [2.], [0.])),
    (2, ([.5], [.25], [0.])),
    (5, ([2], [1], [0.])),
    (8, ([3.5], [1.75], [0.])),
  ]
)
def test_create(pop_num: int, expected_max_avg_min: tuple[list[float], list[float], list[float]]) -> None:
  # values
  number_of_generations=2
  float_iter=(x/2 for x in range(9))
  pop=DummyPop(pop_num, lambda: DummyInd(0, next(float_iter)))

  # test
  gens=G(pop, number_of_generations)

  #results
  # private access: gens.__maxs, gens.__avgs, gens.__mins
  maxs, avgs, mins=gens._Generations__maxs, gens._Generations__avgs, gens._Generations__mins # type: ignore
  # private access: gens.__max_sol, gens.__min_sol
  max_sol, min_sol=gens._Generations__max_sol, gens._Generations__min_sol # type: ignore
  # private access: gens.__max_of_max, gens.__min_of_min
  max_of_max, min_of_min=gens._Generations__max_of_max, gens._Generations__min_of_min # type: ignore

  assert pop.gen==0
  assert gens.curr_generations==0
  assert len(maxs)==number_of_generations+1
  assert len(avgs)==number_of_generations+1
  assert len(mins)==number_of_generations+1
  assert maxs[:1]==pytest.approx(expected_max_avg_min[0])
  assert avgs[:1]==pytest.approx(expected_max_avg_min[1])
  assert mins[:1]==pytest.approx(expected_max_avg_min[2])
  assert isinstance(max_sol, DummyInd)
  assert isinstance(min_sol, DummyInd)
  assert max_sol._gen==pytest.approx((0, expected_max_avg_min[0][0]))
  assert min_sol._gen==pytest.approx((0, expected_max_avg_min[2][0]))
  assert max_of_max==pytest.approx(expected_max_avg_min[0][0])
  assert min_of_min==pytest.approx(expected_max_avg_min[2][0])

@pytest.mark.parametrize(
  ('pop_num', 'expected_max_avg_min'),
  [
    (9, ([4], [2.], [0.])),
    (2, ([.5], [.25], [0.])),
    (5, ([2], [1], [0.])),
    (8, ([3.5], [1.75], [0.])),
  ]
)
def test_get_statistics_on_start(pop_num: int, expected_max_avg_min: tuple[list[float], list[float], list[float]]) -> None:
  # values
  number_of_generations=2
  float_iter=(x/2 for x in range(9))
  pop=DummyPop(pop_num, lambda: DummyInd(0, next(float_iter)))
  gens=G(pop, number_of_generations)

  # test
  ret=gens.get_statistics()

  #results
  assert pop.gen==0
  assert gens.curr_generations==0
  assert isinstance(ret, tuple)
  assert len(ret)==3
  (max_sol, min_sol), (max_of_max, min_of_min), (maxs, avgs, mins)=ret
  assert maxs==pytest.approx(expected_max_avg_min[0])
  assert avgs==pytest.approx(expected_max_avg_min[1])
  assert mins==pytest.approx(expected_max_avg_min[2])
  assert isinstance(max_sol, DummyInd)
  assert isinstance(min_sol, DummyInd)
  assert max_sol._gen==pytest.approx((0, expected_max_avg_min[0][0]))
  assert min_sol._gen==pytest.approx((0, expected_max_avg_min[2][0]))
  assert max_of_max==pytest.approx(expected_max_avg_min[0][0])
  assert min_of_min==pytest.approx(expected_max_avg_min[2][0])

@pytest.mark.parametrize(
  ('go_num_generations', 'pop_num', 'expected_max_avg_min'),
  [
    (2, 2, ([.5, .6, .7], [.25, .35, .45], [.0, .1, .2])),
    (None, 2, ([.5, .6, .7], [.25, .35, .45], [.0, .1, .2])),
    (3, 2, ([.5, .6, .7], [.25, .35, .45], [.0, .1, .2])),
    (2, 9, ([4., 4.1, 4.2], [2., 2.1, 2.2], [.0, .1, .2])),
    (None, 9, ([4., 4.1, 4.2], [2., 2.1, 2.2], [.0, .1, .2])),
    (3, 9, ([4., 4.1, 4.2], [2., 2.1, 2.2], [.0, .1, .2])),
  ]
)
def test_go_through_generations_all_the_way(go_num_generations: int|None, pop_num: int, expected_max_avg_min: tuple[list[float], list[float], list[float]]) -> None:
  # values
  number_of_generations=2
  float_iter=(x/2 for x in range(9))
  pop=DummyPop(pop_num, lambda: DummyInd(0, next(float_iter)))
  gens=G(pop, number_of_generations)

  # test
  gens.go_through_generations(go_num_generations)

  #results
  # private access: gens.__maxs, gens.__avgs, gens.__mins
  maxs, avgs, mins=gens._Generations__maxs, gens._Generations__avgs, gens._Generations__mins # type: ignore
  # private access: gens.__max_sol, gens.__min_sol
  max_sol, min_sol=gens._Generations__max_sol, gens._Generations__min_sol # type: ignore
  # private access: gens.__max_of_max, gens.__min_of_min
  max_of_max, min_of_min=gens._Generations__max_of_max, gens._Generations__min_of_min # type: ignore

  assert pop.gen==number_of_generations
  assert gens.curr_generations==number_of_generations
  assert maxs==pytest.approx(expected_max_avg_min[0])
  assert avgs==pytest.approx(expected_max_avg_min[1])
  assert mins==pytest.approx(expected_max_avg_min[2])
  assert isinstance(max_sol, DummyInd)
  assert isinstance(min_sol, DummyInd)
  assert max_sol._gen==pytest.approx((number_of_generations, max(expected_max_avg_min[0])))
  assert min_sol._gen==pytest.approx((0, min(expected_max_avg_min[2])))
  assert max_of_max==pytest.approx(max(expected_max_avg_min[0]))
  assert min_of_min==pytest.approx(min(expected_max_avg_min[2]))

@pytest.mark.parametrize(
  ('go_num_generations', 'pop_num', 'expected_max_avg_min'),
  [
    (2, 2, ([.5, .6, .7], [.25, .35, .45], [.0, .1, .2])),
    (None, 2, ([.5, .6, .7], [.25, .35, .45], [.0, .1, .2])),
    (3, 2, ([.5, .6, .7], [.25, .35, .45], [.0, .1, .2])),
    (2, 9, ([4., 4.1, 4.2], [2., 2.1, 2.2], [.0, .1, .2])),
    (None, 9, ([4., 4.1, 4.2], [2., 2.1, 2.2], [.0, .1, .2])),
    (3, 9, ([4., 4.1, 4.2], [2., 2.1, 2.2], [.0, .1, .2])),
  ]
)
def test_get_statistics_after_all_the_way(go_num_generations: int|None, pop_num: int, expected_max_avg_min: tuple[list[float], list[float], list[float]]) -> None:
  # values
  number_of_generations=2
  float_iter=(x/2 for x in range(9))
  pop=DummyPop(pop_num, lambda: DummyInd(0, next(float_iter)))
  gens=G(pop, number_of_generations)
  gens.go_through_generations(go_num_generations)

  # test
  ret=gens.get_statistics()

  #results
  assert pop.gen==number_of_generations
  assert gens.curr_generations==number_of_generations
  assert isinstance(ret, tuple)
  assert len(ret)==3
  (max_sol, min_sol), (max_of_max, min_of_min), (maxs, avgs, mins)=ret
  assert maxs==pytest.approx(expected_max_avg_min[0])
  assert avgs==pytest.approx(expected_max_avg_min[1])
  assert mins==pytest.approx(expected_max_avg_min[2])
  assert isinstance(max_sol, DummyInd)
  assert isinstance(min_sol, DummyInd)
  assert max_sol._gen==pytest.approx((number_of_generations, max(expected_max_avg_min[0])))
  assert min_sol._gen==pytest.approx((0, min(expected_max_avg_min[2])))
  assert max_of_max==pytest.approx(max(expected_max_avg_min[0]))
  assert min_of_min==pytest.approx(min(expected_max_avg_min[2]))

@pytest.mark.parametrize(
  ('go_num_generations', 'pop_num', 'expected_max_avg_min'),
  [
    (2, 2, ([.5, .6, .7], [.25, .35, .45], [.0, .1, .2])),
    (1, 2, ([.5, .6], [.25, .35], [.0, .1])),
    (2, 9, ([4., 4.1, 4.2], [2., 2.1, 2.2], [.0, .1, .2])),
    (1, 9, ([4., 4.1], [2., 2.1], [.0, .1])),
  ]
)
def test_go_through_generations_part_way(go_num_generations: int, pop_num: int, expected_max_avg_min: tuple[list[float], list[float], list[float]]) -> None:
  # values
  number_of_generations=3
  float_iter=(x/2 for x in range(9))
  pop=DummyPop(pop_num, lambda: DummyInd(0, next(float_iter)))
  gens=G(pop, number_of_generations)

  # test
  gens.go_through_generations(go_num_generations)

  #results
  # private access: gens.__maxs, gens.__avgs, gens.__mins
  maxs, avgs, mins=gens._Generations__maxs, gens._Generations__avgs, gens._Generations__mins # type: ignore
  # private access: gens.__max_sol, gens.__min_sol
  max_sol, min_sol=gens._Generations__max_sol, gens._Generations__min_sol # type: ignore
  # private access: gens.__max_of_max, gens.__min_of_min
  max_of_max, min_of_min=gens._Generations__max_of_max, gens._Generations__min_of_min # type: ignore

  assert pop.gen==go_num_generations
  assert gens.curr_generations==go_num_generations
  assert maxs[:go_num_generations+1]==pytest.approx(expected_max_avg_min[0])
  assert avgs[:go_num_generations+1]==pytest.approx(expected_max_avg_min[1])
  assert mins[:go_num_generations+1]==pytest.approx(expected_max_avg_min[2])
  assert isinstance(max_sol, DummyInd)
  assert isinstance(min_sol, DummyInd)
  assert max_sol._gen==pytest.approx((go_num_generations, max(expected_max_avg_min[0])))
  assert min_sol._gen==pytest.approx((0, min(expected_max_avg_min[2])))
  assert max_of_max==pytest.approx(max(expected_max_avg_min[0]))
  assert min_of_min==pytest.approx(min(expected_max_avg_min[2]))

@pytest.mark.parametrize(
  ('go_num_generations', 'pop_num', 'expected_max_avg_min'),
  [
    (2, 2, ([.5, .6, .7], [.25, .35, .45], [.0, .1, .2])),
    (1, 2, ([.5, .6], [.25, .35], [.0, .1])),
    (2, 9, ([4., 4.1, 4.2], [2., 2.1, 2.2], [.0, .1, .2])),
    (1, 9, ([4., 4.1], [2., 2.1], [.0, .1])),
  ]
)
def test_get_statistics_after_part_way(go_num_generations: int, pop_num: int, expected_max_avg_min: tuple[list[float], list[float], list[float]]) -> None:
  # values
  number_of_generations=2
  float_iter=(x/2 for x in range(9))
  pop=DummyPop(pop_num, lambda: DummyInd(0, next(float_iter)))
  gens=G(pop, number_of_generations)
  gens.go_through_generations(go_num_generations)

  # test
  ret=gens.get_statistics()

  #results
  assert pop.gen==go_num_generations
  assert gens.curr_generations==go_num_generations
  assert isinstance(ret, tuple)
  assert len(ret)==3
  (max_sol, min_sol), (max_of_max, min_of_min), (maxs, avgs, mins)=ret
  assert maxs==pytest.approx(expected_max_avg_min[0])
  assert avgs==pytest.approx(expected_max_avg_min[1])
  assert mins==pytest.approx(expected_max_avg_min[2])
  assert isinstance(max_sol, DummyInd)
  assert isinstance(min_sol, DummyInd)
  assert max_sol._gen==pytest.approx((go_num_generations, max(expected_max_avg_min[0])))
  assert min_sol._gen==pytest.approx((0, min(expected_max_avg_min[2])))
  assert max_of_max==pytest.approx(max(expected_max_avg_min[0]))
  assert min_of_min==pytest.approx(min(expected_max_avg_min[2]))

@pytest.mark.parametrize(
  ('go_num_generations', 'pop_num', 'expected_max_avg_min'),
  [
    (2, 2, ([.5, .6, .7, .8], [.25, .35, .45, .55], [.0, .1, .2, .3])),
    (None, 2, ([.5, .6, .7, .8], [.25, .35, .45, .55], [.0, .1, .2, .3])),
    (1, 2, ([.5, .6, .7], [.25, .35, .45], [.0, .1, .2])),
    (2, 9, ([4., 4.1, 4.2, 4.3], [2., 2.1, 2.2, 2.3], [.0, .1, .2, .3])),
    (None, 9, ([4., 4.1, 4.2, 4.3], [2., 2.1, 2.2, 2.3], [.0, .1, .2, .3])),
    (1, 9, ([4., 4.1, 4.2], [2., 2.1, 2.2], [.0, .1, .2])),
  ]
)
def test_go_through_generations_after_going_part_way(go_num_generations: int|None, pop_num: int, expected_max_avg_min: tuple[list[float], list[float], list[float]]) -> None:
  # values
  number_of_generations=3
  float_iter=(x/2 for x in range(9))
  pop=DummyPop(pop_num, lambda: DummyInd(0, next(float_iter)))
  gens=G(pop, number_of_generations)
  gens.go_through_generations(1)

  # test
  gens.go_through_generations(go_num_generations)

  #results
  # private access: gens.__maxs, gens.__avgs, gens.__mins
  maxs, avgs, mins=gens._Generations__maxs, gens._Generations__avgs, gens._Generations__mins # type: ignore
  # private access: gens.__max_sol, gens.__min_sol
  max_sol, min_sol=gens._Generations__max_sol, gens._Generations__min_sol # type: ignore
  # private access: gens.__max_of_max, gens.__min_of_min
  max_of_max, min_of_min=gens._Generations__max_of_max, gens._Generations__min_of_min # type: ignore

  if go_num_generations is None:
    go_num_generations=number_of_generations-1
  assert pop.gen==go_num_generations+1
  assert gens.curr_generations==go_num_generations+1
  assert maxs[:go_num_generations+2]==pytest.approx(expected_max_avg_min[0])
  assert avgs[:go_num_generations+2]==pytest.approx(expected_max_avg_min[1])
  assert mins[:go_num_generations+2]==pytest.approx(expected_max_avg_min[2])
  assert isinstance(max_sol, DummyInd)
  assert isinstance(min_sol, DummyInd)
  assert max_sol._gen==pytest.approx((go_num_generations+1, max(expected_max_avg_min[0])))
  assert min_sol._gen==pytest.approx((0, min(expected_max_avg_min[2])))
  assert max_of_max==pytest.approx(max(expected_max_avg_min[0]))
  assert min_of_min==pytest.approx(min(expected_max_avg_min[2]))

@pytest.mark.parametrize(
  ('go_num_generations', 'pop_num', 'expected_max_avg_min'),
  [
    (2, 2, ([.5, .6, .7, .8], [.25, .35, .45, .55], [.0, .1, .2, .3])),
    (None, 2, ([.5, .6, .7, .8], [.25, .35, .45, .55], [.0, .1, .2, .3])),
    (1, 2, ([.5, .6, .7], [.25, .35, .45], [.0, .1, .2])),
    (2, 9, ([4., 4.1, 4.2, 4.3], [2., 2.1, 2.2, 2.3], [.0, .1, .2, .3])),
    (None, 9, ([4., 4.1, 4.2, 4.3], [2., 2.1, 2.2, 2.3], [.0, .1, .2, .3])),
    (1, 9, ([4., 4.1, 4.2], [2., 2.1, 2.2], [.0, .1, .2])),
  ]
)
def test_get_statistics_after_part_way_after_going_part_way(go_num_generations: int|None, pop_num: int, expected_max_avg_min: tuple[list[float], list[float], list[float]]) -> None:
  # values
  number_of_generations=3
  float_iter=(x/2 for x in range(9))
  pop=DummyPop(pop_num, lambda: DummyInd(0, next(float_iter)))
  gens=G(pop, number_of_generations)
  gens.go_through_generations(1)
  gens.go_through_generations(go_num_generations)

  # test
  ret=gens.get_statistics()

  #results
  if go_num_generations is None:
    go_num_generations=number_of_generations-1
  assert pop.gen==go_num_generations+1
  assert gens.curr_generations==go_num_generations+1
  assert isinstance(ret, tuple)
  assert len(ret)==3
  (max_sol, min_sol), (max_of_max, min_of_min), (maxs, avgs, mins)=ret
  assert maxs==pytest.approx(expected_max_avg_min[0])
  assert avgs==pytest.approx(expected_max_avg_min[1])
  assert mins==pytest.approx(expected_max_avg_min[2])
  assert isinstance(max_sol, DummyInd)
  assert isinstance(min_sol, DummyInd)
  assert max_sol._gen==pytest.approx((go_num_generations+1, max(expected_max_avg_min[0])))
  assert min_sol._gen==pytest.approx((0, min(expected_max_avg_min[2])))
  assert max_of_max==pytest.approx(max(expected_max_avg_min[0]))
  assert min_of_min==pytest.approx(min(expected_max_avg_min[2]))

@pytest.mark.parametrize(
  ('go_num_generations', 'pop_num', 'expected_max_avg_min'),
  [
    ((1,), 2, ([.5, .6, .7, .8], [.25, .35, .45, .55], [.0, .1, .2, .3])),
    ((2,), 2, ([.5, .6, .7, .8, .9], [.25, .35, .45, .55, .65], [.0, .1, .2, .3, .4])),
    ((1, 1), 2, ([.5, .6, .7, .8, .9], [.25, .35, .45, .55, .65], [.0, .1, .2, .3, .4])),
    ((1, 2), 2, ([.5, .6, .7, .8, .9, 1.], [.25, .35, .45, .55, .65, .75], [.0, .1, .2, .3, .4, .5])),
    ((2, 1), 2, ([.5, .6, .7, .8, .9, 1.], [.25, .35, .45, .55, .65, .75], [.0, .1, .2, .3, .4, .5])),
    ((1, 1, 1), 2, ([.5, .6, .7, .8, .9, 1.], [.25, .35, .45, .55, .65, .75], [.0, .1, .2, .3, .4, .5])),
    ((3,), 2, ([.5, .6, .7, .8, .9, 1.], [.25, .35, .45, .55, .65, .75], [.0, .1, .2, .3, .4, .5])),
    ((1,), 9, ([4., 4.1, 4.2, 4.3], [2., 2.1, 2.2, 2.3], [.0, .1, .2, .3])),
    ((2,), 9, ([4., 4.1, 4.2, 4.3, 4.4], [2., 2.1, 2.2, 2.3, 2.4], [.0, .1, .2, .3, .4])),
    ((1, 1), 9, ([4., 4.1, 4.2, 4.3, 4.4], [2., 2.1, 2.2, 2.3, 2.4], [.0, .1, .2, .3, .4])),
    ((1, 2), 9, ([4., 4.1, 4.2, 4.3, 4.4, 4.5], [2., 2.1, 2.2, 2.3, 2.4, 2.5], [.0, .1, .2, .3, .4, .5])),
    ((2, 1), 9, ([4., 4.1, 4.2, 4.3, 4.4, 4.5], [2., 2.1, 2.2, 2.3, 2.4, 2.5], [.0, .1, .2, .3, .4, .5])),
    ((1, 1, 1), 9, ([4., 4.1, 4.2, 4.3, 4.4, 4.5], [2., 2.1, 2.2, 2.3, 2.4, 2.5], [.0, .1, .2, .3, .4, .5])),
    ((3,), 9, ([4., 4.1, 4.2, 4.3, 4.4, 4.5], [2., 2.1, 2.2, 2.3, 2.4, 2.5], [.0, .1, .2, .3, .4, .5])),
  ]
)
def test_go_through_generations_multiple(go_num_generations: tuple[int, ...], pop_num: int, expected_max_avg_min: tuple[list[float], list[float], list[float]]) -> None:
  # values
  number_of_generations=5
  float_iter=(x/2 for x in range(9))
  pop=DummyPop(pop_num, lambda: DummyInd(0, next(float_iter)))
  gens=G(pop, number_of_generations)
  gens.go_through_generations(1)
  gens.go_through_generations(1)

  # test
  for i in go_num_generations:
    gens.go_through_generations(i)

  #results
  # private access: gens.__maxs, gens.__avgs, gens.__mins
  maxs, avgs, mins=gens._Generations__maxs, gens._Generations__avgs, gens._Generations__mins # type: ignore
  # private access: gens.__max_sol, gens.__min_sol
  max_sol, min_sol=gens._Generations__max_sol, gens._Generations__min_sol # type: ignore
  # private access: gens.__max_of_max, gens.__min_of_min
  max_of_max, min_of_min=gens._Generations__max_of_max, gens._Generations__min_of_min # type: ignore

  assert pop.gen==sum(go_num_generations)+2
  assert gens.curr_generations==sum(go_num_generations)+2
  assert maxs[:sum(go_num_generations)+3]==pytest.approx(expected_max_avg_min[0])
  assert avgs[:sum(go_num_generations)+3]==pytest.approx(expected_max_avg_min[1])
  assert mins[:sum(go_num_generations)+3]==pytest.approx(expected_max_avg_min[2])
  assert isinstance(max_sol, DummyInd)
  assert isinstance(min_sol, DummyInd)
  assert max_sol._gen==pytest.approx((sum(go_num_generations)+2, max(expected_max_avg_min[0])))
  assert min_sol._gen==pytest.approx((0, min(expected_max_avg_min[2])))
  assert max_of_max==pytest.approx(max(expected_max_avg_min[0]))
  assert min_of_min==pytest.approx(min(expected_max_avg_min[2]))

@pytest.mark.parametrize(
  ('go_num_generations', 'pop_num', 'expected_max_avg_min'),
  [
    ((1,), 2, ([.5, .6, .7, .8], [.25, .35, .45, .55], [.0, .1, .2, .3])),
    ((2,), 2, ([.5, .6, .7, .8, .9], [.25, .35, .45, .55, .65], [.0, .1, .2, .3, .4])),
    ((1, 1), 2, ([.5, .6, .7, .8, .9], [.25, .35, .45, .55, .65], [.0, .1, .2, .3, .4])),
    ((1, 2), 2, ([.5, .6, .7, .8, .9, 1.], [.25, .35, .45, .55, .65, .75], [.0, .1, .2, .3, .4, .5])),
    ((2, 1), 2, ([.5, .6, .7, .8, .9, 1.], [.25, .35, .45, .55, .65, .75], [.0, .1, .2, .3, .4, .5])),
    ((1, 1, 1), 2, ([.5, .6, .7, .8, .9, 1.], [.25, .35, .45, .55, .65, .75], [.0, .1, .2, .3, .4, .5])),
    ((3,), 2, ([.5, .6, .7, .8, .9, 1.], [.25, .35, .45, .55, .65, .75], [.0, .1, .2, .3, .4, .5])),
    ((1,), 9, ([4., 4.1, 4.2, 4.3], [2., 2.1, 2.2, 2.3], [.0, .1, .2, .3])),
    ((2,), 9, ([4., 4.1, 4.2, 4.3, 4.4], [2., 2.1, 2.2, 2.3, 2.4], [.0, .1, .2, .3, .4])),
    ((1, 1), 9, ([4., 4.1, 4.2, 4.3, 4.4], [2., 2.1, 2.2, 2.3, 2.4], [.0, .1, .2, .3, .4])),
    ((1, 2), 9, ([4., 4.1, 4.2, 4.3, 4.4, 4.5], [2., 2.1, 2.2, 2.3, 2.4, 2.5], [.0, .1, .2, .3, .4, .5])),
    ((2, 1), 9, ([4., 4.1, 4.2, 4.3, 4.4, 4.5], [2., 2.1, 2.2, 2.3, 2.4, 2.5], [.0, .1, .2, .3, .4, .5])),
    ((1, 1, 1), 9, ([4., 4.1, 4.2, 4.3, 4.4, 4.5], [2., 2.1, 2.2, 2.3, 2.4, 2.5], [.0, .1, .2, .3, .4, .5])),
    ((3,), 9, ([4., 4.1, 4.2, 4.3, 4.4, 4.5], [2., 2.1, 2.2, 2.3, 2.4, 2.5], [.0, .1, .2, .3, .4, .5])),
  ]
)
def test_get_statistics_after_multiple(go_num_generations: tuple[int, ...], pop_num: int, expected_max_avg_min: tuple[list[float], list[float], list[float]]) -> None:
  # values
  number_of_generations=5
  float_iter=(x/2 for x in range(9))
  pop=DummyPop(pop_num, lambda: DummyInd(0, next(float_iter)))
  gens=G(pop, number_of_generations)
  gens.go_through_generations(1)
  gens.go_through_generations(1)
  for i in go_num_generations:
    gens.go_through_generations(i)

  # test
  ret=gens.get_statistics()

  #results
  assert pop.gen==sum(go_num_generations)+2
  assert gens.curr_generations==sum(go_num_generations)+2
  assert isinstance(ret, tuple)
  assert len(ret)==3
  (max_sol, min_sol), (max_of_max, min_of_min), (maxs, avgs, mins)=ret
  assert maxs==pytest.approx(expected_max_avg_min[0])
  assert avgs==pytest.approx(expected_max_avg_min[1])
  assert mins==pytest.approx(expected_max_avg_min[2])
  assert isinstance(max_sol, DummyInd)
  assert isinstance(min_sol, DummyInd)
  assert max_sol._gen==pytest.approx((sum(go_num_generations)+2, max(expected_max_avg_min[0])))
  assert min_sol._gen==pytest.approx((0, min(expected_max_avg_min[2])))
  assert max_of_max==pytest.approx(max(expected_max_avg_min[0]))
  assert min_of_min==pytest.approx(min(expected_max_avg_min[2]))

@pytest.mark.parametrize(
  ('go_num_generations', 'pop_num', 'expected_max_avg_min'),
  [(g, pn, emam) for g, (pn, emam) in itertools.product((
    (1, None),
    (2, None),
    (1, 1, None),
    (None,),
    (2, 4,),
    (2, 3,),
    (2, 2, 1),
    (2, 1, 2),
    (2, 1, 1, 1),
  ),(
    (2, ([.5, .6, .7, .8, .9, 1.], [.25, .35, .45, .55, .65, .75], [.0, .1, .2, .3, .4, .5])),
    (9, ([4., 4.1, 4.2, 4.3, 4.4, 4.5], [2., 2.1, 2.2, 2.3, 2.4, 2.5], [.0, .1, .2, .3, .4, .5])),
  ))]
)
def test_error_go_through_generations_after_going_to_the_end(go_num_generations: tuple[int|None, ...], pop_num: int, expected_max_avg_min: tuple[list[float], list[float], list[float]]) -> None:
  # values
  number_of_generations=5
  float_iter=(x/2 for x in range(9))
  pop=DummyPop(pop_num, lambda: DummyInd(0, next(float_iter)))
  gens=G(pop, number_of_generations)
  for i in go_num_generations:
    gens.go_through_generations(i)

  # test
  with pytest.raises(IndexError) as excinfo:
    gens.go_through_generations(1)

  # results
  assert str(excinfo.value)=='Tried to add next generation(s) after reaching max number of them'
  # private access: gens.__maxs, gens.__avgs, gens.__mins
  maxs, avgs, mins=gens._Generations__maxs, gens._Generations__avgs, gens._Generations__mins # type: ignore
  # private access: gens.__max_sol, gens.__min_sol
  max_sol, min_sol=gens._Generations__max_sol, gens._Generations__min_sol # type: ignore
  # private access: gens.__max_of_max, gens.__min_of_min
  max_of_max, min_of_min=gens._Generations__max_of_max, gens._Generations__min_of_min # type: ignore

  assert pop.gen==number_of_generations
  assert gens.curr_generations==number_of_generations
  assert maxs==pytest.approx(expected_max_avg_min[0])
  assert avgs==pytest.approx(expected_max_avg_min[1])
  assert mins==pytest.approx(expected_max_avg_min[2])
  assert isinstance(max_sol, DummyInd)
  assert isinstance(min_sol, DummyInd)
  assert max_sol._gen==pytest.approx((number_of_generations, max(expected_max_avg_min[0])))
  assert min_sol._gen==pytest.approx((0, min(expected_max_avg_min[2])))
  assert max_of_max==pytest.approx(max(expected_max_avg_min[0]))
  assert min_of_min==pytest.approx(min(expected_max_avg_min[2]))

@pytest.mark.parametrize(
  'number_of_generations',
  [0, -1, -2]
)
def test_invalid_num_gen(number_of_generations: int) -> None:
  # values
  float_iter=range(9)
  pop_num=len(float_iter)
  float_iter=(x/2 for x in float_iter)
  pop=DummyPop(pop_num, lambda: DummyInd(0, next(float_iter)))

  # test
  with pytest.raises(ValueError) as excinfo:
    G(pop, number_of_generations)

  # results
  assert str(excinfo.value)=='max_num_gen: is too small, it should at least equal 1'
