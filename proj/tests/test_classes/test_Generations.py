import pytest
import itertools
import typing as t
from pathlib import Path
from tqdm.auto import tqdm
from gaaml.classes.Generations import Generations as G
from gaaml.classes.Population import Population as _P
from gaaml.classes.Individual import _BaseIndividual as _BI
# from gaaml.classes.GenIndividual import GenIndividual as _GI

class DummyInd(_BI["DummyInd", int, tuple[int, float], float]):
  def __init__(self, gen_int: int, gen_float: float) -> None:
    super().__init__((gen_int, gen_float), lambda gen: gen[1])
  def mutate(self) -> None: ...

  _DI=t.TypeVar('_DI', bound="DummyInd")
  @classmethod
  def get_cp(cls: type[_DI], a: _DI, b: _DI) -> int:
    return 0
  @classmethod
  def crossover(cls: type[_DI], a: _DI, b: _DI, cp: int) -> tuple[_DI, _DI]:
    (a_gen0, a_gen1)=a._gen
    (b_gen0, b_gen1)=b._gen
    return cls(a_gen0+1, a_gen1+.1), cls(b_gen0+1, b_gen1+.1)
  def _save_format(self) -> dict[str, object]:
    return {
      'name': self.__class__.__name__,
      'gen': self._gen,
    }

class DummyPop(_P[DummyInd]):
  gen_num: int=0
  def __init__(
    self,
    pop_num: int,
    fitnesses_progress_output: tqdm,
    individual_factory: t.Callable[[], DummyInd],
  ) -> None:
    calc_fitness_func: t.Callable[[DummyInd, Path], float]=lambda ind, dir: ind.gen
    super().__init__(
      pop_num,
      individual_factory,
      calc_fitness_func,
      1.,
      .0,
      fitnesses_progress_output=fitnesses_progress_output,
      max_worker_num=1,
    )
  def next_generation(self, gen_num: int) -> None:
    self.gen_num+=1
    assert self.gen_num==gen_num
    self.__selection_list=iter(self.population)
    return super().next_generation(gen_num)
  def _selection(self) -> tuple[DummyInd, DummyInd]:
    ind1=next(self.__selection_list)
    # if self.__selection_list.
    try:
      ind2=next(self.__selection_list)
    except StopIteration:
      ind2=self.population[0]
    return ind1, ind2

def bar_asserts(bar: tqdm, n: int, *, close: bool=True) -> None:
  assert bar.n==n
  assert not bar.disable
  if close:
    bar.close()

def get_priv_pop(
  gens: G[[int, tqdm, t.Callable[[], DummyInd]], DummyInd],
) -> DummyPop:
  pop=gens._Generations__pop # type: ignore
  return pop
def num_generations_asserts(
  gens: G[[int, tqdm, t.Callable[[], DummyInd]], DummyInd],
  *,
  curr_generation: int,
) -> None:
  pop=get_priv_pop(gens)
  assert pop.gen_num==curr_generation
  assert gens.curr_generations==curr_generation

def get_privates(
  gens: G[[int, tqdm, t.Callable[[], DummyInd]], DummyInd],
) -> tuple[
  list[float],
  list[float],
  list[float],
  DummyInd,
  DummyInd,
  float,
  float,
]:
  maxs=gens._Generations__maxs # type: ignore
  avgs=gens._Generations__avgs # type: ignore
  mins=gens._Generations__mins # type: ignore
  max_sol=gens._Generations__max_sol # type: ignore
  min_sol=gens._Generations__min_sol # type: ignore
  max_of_max=gens._Generations__max_of_max # type: ignore
  min_of_min=gens._Generations__min_of_min # type: ignore
  return (
    maxs,
    avgs,
    mins,
    max_sol,
    min_sol,
    max_of_max,
    min_of_min,
  )

def generations_asserts(
  gens: G[[int, tqdm, t.Callable[[], DummyInd]], DummyInd],
  *,
  curr_generation: int,
  expected_max_avg_min: tuple[
    list[float],
    list[float],
    list[float],
  ],
  expected_len: int|None=None,
) -> None:
  (
    maxs,
    avgs,
    mins,
    max_sol,
    min_sol,
    max_of_max,
    min_of_min,
  )=get_privates(gens)
  exp_maxs, exp_avgs, exp_mins=expected_max_avg_min
  num_generations_asserts(gens, curr_generation=curr_generation)
  if expected_len is not None:
    assert len(maxs)==expected_len
    assert len(avgs)==expected_len
    assert len(mins)==expected_len
  assert maxs[:curr_generation+1]==pytest.approx(exp_maxs)
  assert avgs[:curr_generation+1]==pytest.approx(exp_avgs)
  assert mins[:curr_generation+1]==pytest.approx(exp_mins)
  assert isinstance(max_sol, DummyInd)
  assert isinstance(min_sol, DummyInd)
  assert max_sol._gen==pytest.approx((
    curr_generation,
    max(exp_maxs),
  ))
  assert min_sol._gen==pytest.approx((
    0,
    min(exp_mins),
  ))
  assert max_of_max==pytest.approx(max(exp_maxs))
  assert min_of_min==pytest.approx(min(exp_mins))

def statistics_asserts(
  ret: tuple[
    tuple[DummyInd, DummyInd],
    tuple[float, float],
    tuple[list[float], list[float], list[float]],
  ],
  *,
  curr_generation: int,
  expected_max_avg_min: tuple[
    list[float],
    list[float],
    list[float],
  ],
) -> None:
  exp_maxs, exp_avgs, exp_mins=expected_max_avg_min
  exp_max_of_max, exp_min_of_min=max(exp_maxs), min(exp_mins)

  assert isinstance(ret, tuple)
  assert len(ret)==3
  (
    (max_sol, min_sol),
    (max_of_max, min_of_min),
    (maxs, avgs, mins),
  )=ret
  assert maxs==pytest.approx(exp_maxs)
  assert avgs==pytest.approx(exp_avgs)
  assert mins==pytest.approx(exp_mins)
  assert isinstance(max_sol, DummyInd)
  assert isinstance(min_sol, DummyInd)
  assert max_sol._gen==pytest.approx((
    curr_generation,
    exp_max_of_max,
  ))
  assert min_sol._gen==pytest.approx((
    0,
    exp_min_of_min,
  ))
  assert max_of_max==pytest.approx(exp_max_of_max)
  assert min_of_min==pytest.approx(exp_min_of_min)

def create_gens(
  number_of_generations: int,
  pop_num: int,
) -> tuple[
  G[[int, tqdm, t.Callable[[], DummyInd]], DummyInd],
  tqdm,
  tqdm,
]:
  float_iter=(x/2 for x in range(9))
  bar1=tqdm(
    total=number_of_generations,
    desc='Generations',
    position=0,
    mininterval=0,
  )
  bar2=tqdm(
    total=pop_num,
    desc='Calculated fitnesses',
    position=1,
    mininterval=0,
  )
  gens=G(
    number_of_generations,
    bar1,
    DummyPop,
    pop_num,
    bar2,
    lambda: DummyInd(0, next(float_iter)),
  )
  return gens, bar1, bar2

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
  bar1=tqdm(total=number_of_generations, desc='Generations', position=0, mininterval=0)
  bar2=tqdm(total=pop_num, desc='Calculated fitnesses', position=1, mininterval=0)
  pop_args, pop_kwargs=(pop_num, bar2, lambda: DummyInd(0, next(float_iter))), {}

  # test
  gens=G(number_of_generations, bar1, DummyPop, *pop_args, **pop_kwargs)

  # results
  generations_asserts(
    gens,
    curr_generation=0,
    expected_max_avg_min=expected_max_avg_min,
    expected_len=number_of_generations+1,
  )
  bar_asserts(bar2, pop_num)
  bar_asserts(bar1, gens.curr_generations)

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
  gens, bar1, bar2=create_gens(number_of_generations, pop_num)

  # test
  ret=gens.get_statistics()

  # results
  num_generations_asserts(gens, curr_generation=0)
  statistics_asserts(
    ret,
    curr_generation=0,
    expected_max_avg_min=expected_max_avg_min,
  )
  bar_asserts(bar2, pop_num)
  bar_asserts(bar1, gens.curr_generations)

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
  gens, bar1, bar2=create_gens(number_of_generations, pop_num)

  # test
  gens.go_through_generations(go_num_generations)

  # results
  generations_asserts(
    gens,
    curr_generation=number_of_generations,
    expected_max_avg_min=expected_max_avg_min,
  )
  bar_asserts(bar2, pop_num)
  bar_asserts(bar1, gens.curr_generations)

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
  gens, bar1, bar2=create_gens(number_of_generations, pop_num)
  gens.go_through_generations(go_num_generations)

  # test
  ret=gens.get_statistics()

  # results
  num_generations_asserts(gens, curr_generation=number_of_generations)
  statistics_asserts(
    ret,
    curr_generation=number_of_generations,
    expected_max_avg_min=expected_max_avg_min,
  )
  bar_asserts(bar2, pop_num)
  bar_asserts(bar1, gens.curr_generations)

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
  gens, bar1, bar2=create_gens(number_of_generations, pop_num)

  # test
  gens.go_through_generations(go_num_generations)

  # results
  generations_asserts(
    gens,
    curr_generation=go_num_generations,
    expected_max_avg_min=expected_max_avg_min,
  )
  bar_asserts(bar2, pop_num)
  bar_asserts(bar1, gens.curr_generations)

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
  number_of_generations=3
  gens, bar1, bar2=create_gens(number_of_generations, pop_num)
  gens.go_through_generations(go_num_generations)

  # test
  ret=gens.get_statistics()

  # results
  num_generations_asserts(gens, curr_generation=go_num_generations)
  statistics_asserts(
    ret,
    curr_generation=go_num_generations,
    expected_max_avg_min=expected_max_avg_min,
  )
  bar_asserts(bar2, pop_num)
  bar_asserts(bar1, gens.curr_generations)

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
  gens, bar1, bar2=create_gens(number_of_generations, pop_num)
  gens.go_through_generations(1)

  # test
  gens.go_through_generations(go_num_generations)

  # results
  if go_num_generations is None:
    go_num_generations=number_of_generations-1
  generations_asserts(
    gens,
    curr_generation=go_num_generations+1,
    expected_max_avg_min=expected_max_avg_min,
  )
  bar_asserts(bar2, pop_num)
  bar_asserts(bar1, gens.curr_generations)

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
  gens, bar1, bar2=create_gens(number_of_generations, pop_num)
  gens.go_through_generations(1)
  gens.go_through_generations(go_num_generations)

  # test
  ret=gens.get_statistics()

  # results
  if go_num_generations is None:
    go_num_generations=number_of_generations-1
  curr_generation=go_num_generations+1
  num_generations_asserts(gens, curr_generation=curr_generation)
  statistics_asserts(
    ret,
    curr_generation=curr_generation,
    expected_max_avg_min=expected_max_avg_min,
  )
  bar_asserts(bar2, pop_num)
  bar_asserts(bar1, gens.curr_generations)

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
  gens, bar1, bar2=create_gens(number_of_generations, pop_num)
  gens.go_through_generations(1)
  gens.go_through_generations(1)

  # test
  for i in go_num_generations:
    gens.go_through_generations(i)

  # results
  generations_asserts(
    gens,
    curr_generation=sum(go_num_generations)+2,
    expected_max_avg_min=expected_max_avg_min,
  )
  bar_asserts(bar2, pop_num)
  bar_asserts(bar1, gens.curr_generations)

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
  gens, bar1, bar2=create_gens(number_of_generations, pop_num)
  gens.go_through_generations(1)
  gens.go_through_generations(1)
  for i in go_num_generations:
    gens.go_through_generations(i)

  # test
  ret=gens.get_statistics()

  # results
  curr_generation=sum(go_num_generations)+2
  num_generations_asserts(gens, curr_generation=curr_generation)
  statistics_asserts(
    ret,
    curr_generation=curr_generation,
    expected_max_avg_min=expected_max_avg_min,
  )
  bar_asserts(bar2, pop_num)
  bar_asserts(bar1, gens.curr_generations)

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
  gens, bar1, bar2=create_gens(number_of_generations, pop_num)
  for i in go_num_generations:
    gens.go_through_generations(i)

  # test
  with pytest.raises(IndexError) as excinfo:
    gens.go_through_generations(1)

  # results
  assert str(excinfo.value)=='Tried to add next generation(s) after reaching max number of them'
  generations_asserts(
    gens,
    curr_generation=number_of_generations,
    expected_max_avg_min=expected_max_avg_min,
  )
  bar_asserts(bar2, pop_num)
  bar_asserts(bar1, gens.curr_generations)

@pytest.mark.parametrize(
  'number_of_generations',
  [0, -1, -2]
)
def test_invalid_num_gen(number_of_generations: int) -> None:
  # values
  float_iter=range(9)
  pop_num=len(float_iter)
  float_iter=(x/2 for x in float_iter)
  bar1=tqdm(total=number_of_generations, desc='Generations', position=0, mininterval=0)
  bar2=tqdm(total=pop_num, desc='Calculated fitnesses', position=1, mininterval=0)
  pop_args, pop_kwargs=(pop_num, bar2, lambda: DummyInd(0, next(float_iter))), {}

  # test
  with pytest.raises(ValueError) as excinfo:
    _=G(number_of_generations, bar1, DummyPop, *pop_args, **pop_kwargs)

  # results
  assert str(excinfo.value)=='max_num_gen: is too small, it should at least equal 1'
  bar_asserts(bar2, 0)
  bar_asserts(bar1, 0)
