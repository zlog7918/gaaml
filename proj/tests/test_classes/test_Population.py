import pytest
import typing as t
import random as rnd
from gaaml.classes import _utils as util
from gaaml.classes.MaxAvgMinHolder import MaxAvgMinHolder as MAMH
from gaaml.classes.Population import Population as P
from gaaml.classes.GenIndividual import GenIndividual as _GI


@pytest.mark.parametrize(
  ('seed', 'flag'),
  [*zip(range(10), (
    False, # 0.8444218515250481
    True, # 0.13436424411240122
    False, # 0.9560342718892494
    True, # 0.23796462709189137
    True, # 0.23604808973743452
    True, # 0.6229016948897019
    False, # 0.793340083761663
    True, # 0.32383276483316237
    True, # 0.2267058593810488
    True, # 0.46300735781502145
  ))]
)
def test_crossover(seed: int, flag: bool) -> None:
  # values
  input_len=4
  gi1=_GI(input_len)
  gi2=_GI(input_len)
  crossover_rate=.77
  rnd.seed(seed)

  # test
  ret=P.crossover(gi1, gi2, crossover_rate)

  # result
  assert isinstance(ret, tuple)
  cgi1, cgi2=ret
  assert isinstance(cgi1, _GI)
  assert isinstance(cgi2, _GI)
  assert (cgi1 is gi1)^flag
  assert (cgi2 is gi2)^flag

@pytest.mark.parametrize(
  ('seed', 'flag'),
  [*zip(range(20), (
    False, # 0.8444218515250481
    True, # 0.13436424411240122
    False, # 0.9560342718892494
    False, # 0.23796462709189137
    False, # 0.23604808973743452
    False, # 0.6229016948897019
    False, # 0.793340083761663
    False, # 0.32383276483316237
    False, # 0.2267058593810488
    False, # 0.46300735781502145
    False, # 0.5714025946899135
    False, # 0.4523795535098186
    False, # 0.4745706786885481
    False, # 0.2590084917154736
    True, # 0.10682853770165568
    False, # 0.965242141552123
    False, # 0.36152277491407514
    False, # 0.5219839097124932
    True, # 0.18126486333322134
    False, # 0.6771258268002703
  ))]
)
def test_mutate(seed: int, flag: bool) -> None:
  # values
  input_len=4
  gi=_GI(input_len)
  mutation_rate=.2
  oryg_gen=gi.gen[:]
  rnd.seed(seed)

  # test
  mgi=P.mutate(gi, mutation_rate)

  # result
  assert isinstance(mgi, _GI)
  assert mgi is gi
  assert (mgi.gen==oryg_gen)!=flag

@pytest.mark.parametrize(
  ('values', 'exp_to_add'),
  [
    ((0, 1), .01),
    ((1, 4), .01),
    ((8, 4, 3), .03),
    ((140, 510, 753), 1.4),
  ]
)
def test_calc_to_add(values: tuple[float, ...], exp_to_add: float) -> None:
  # values
  handle=MAMH(5)
  for v in values:
    handle.append(v)

  # test
  to_add=P._calc_to_add(handle)

  # result
  assert isinstance(float(to_add), float)
  assert to_add==pytest.approx(exp_to_add)

def test_create() -> None:
  # values
  pop_num=5
  bit,min_v,max_v=4,1,11
  input_len=bit
  def __cr_ind() -> t.Generator[_GI, None, None]:
    for gi_gen in (
      '1000', # fit: 8+1=9
      '0011', # fit: 3+1=4
      '0100', # fit: 4+1=5
      '0110', # fit: 6+1=7
      '1101', # fit: 13+1>11 -> 9+1=10
    ):
      gi=_GI(input_len)
      gi._gen=bytearray(gi_gen.encode())
      yield gi
  gi1, gi2, gi3, gi4, gi5=(x for x in __cr_ind())
  _cr_ind=(x for x in (gi1, gi2, gi3, gi4, gi5))
  cr_ind: t.Callable[[], _GI]=lambda: next(_cr_ind)
  def calc_fitness_func(ind: _GI) -> float:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v)
  crossover_rate=.8
  mutation_rate=.1
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  exp_fitnesses=(9,4,5,7,10)

  # test
  pop=P(*schema)

  # results
  assert isinstance(pop.population, list)
  assert isinstance(pop.fitnesses, list)

  assert len(pop.population)==pop_num
  assert len(pop.fitnesses)==pop_num
  assert all(isinstance(ind, _GI) for ind in pop.population)
  assert all(len(ind.gen)==input_len for ind in pop.population)
  assert all(ind is gi for ind, gi in zip(pop.population, (gi1, gi2, gi3, gi4, gi5)))
  assert all(isinstance(float(fit), float) for fit in pop.fitnesses)
  assert all(
    min_v<=fit
    and fit<=max_v
    and fit==exp_fit
      for fit, exp_fit in
    zip(pop.fitnesses, exp_fitnesses)
  )

def test_population_returns_copy() -> None:
  # values
  pop_num=2
  bit,min_v,max_v=4,1,11
  input_len=bit
  cr_ind: t.Callable[[], _GI]=lambda: _GI(input_len)
  def calc_fitness_func(ind: _GI) -> float:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v)
  crossover_rate=.8
  mutation_rate=.1
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema)

  # test
  population_copy=pop.population
  population_copy.append(cr_ind())

  # results
  assert len(pop.population)==pop_num

def test_get_max_avg_min() -> None:
  # values
  pop_num=5
  bit,min_v,max_v=4,1,11
  input_len=bit
  def __cr_ind() -> t.Generator[_GI, None, None]:
    for gi_gen in (
      '1000', # fit: 8+1=9
      '0011', # fit: 3+1=4
      '0100', # fit: 4+1=5
      '0110', # fit: 6+1=7
      '1101', # fit: 13+1>11 -> 9+1=10
    ):
      gi=_GI(input_len)
      gi._gen=bytearray(gi_gen.encode())
      yield gi
  gi1, gi2, gi3, gi4, gi5=(x for x in __cr_ind())
  _cr_ind=(x for x in (gi1, gi2, gi3, gi4, gi5))
  cr_ind: t.Callable[[], _GI]=lambda: next(_cr_ind)
  def calc_fitness_func(ind: _GI) -> float:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v)
  crossover_rate=.8
  mutation_rate=.1
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  exp_fitnesses=(9,4,5,7,10)
  pop=P(*schema)

  # test
  ret=pop.get_max_avg_min()

  # results
  assert isinstance(ret, tuple)
  assert len(ret)==5
  max_ind, min_ind, max_f, avg_f, min_f=ret
  assert isinstance(max_ind, _GI)
  assert isinstance(min_ind, _GI)
  assert isinstance(float(max_f), float)
  assert max_f==max(exp_fitnesses)
  assert isinstance(float(avg_f), float)
  assert avg_f==sum(exp_fitnesses)/5
  assert isinstance(float(min_f), float)  
  assert min_f==min(exp_fitnesses)

def test_selection_sum_zero() -> None:
  # values
  pop_num=5
  bit,min_v,max_v=4,1,11
  input_len=bit
  def calc_fitness_func(ind: _GI) -> float:
    ret=util.correct_gen_to_min_max(ind.gen, min_v, max_v)
    ret-=7
    return 0 if ret<0 else ret
  def __cr_ind() -> t.Generator[_GI]:
    for gi_gen in (
      '0100', # fit: 4+1=5 -> 5-7=-2 -> 0
      '0011', # fit: 3+1=4 -> 4-7=-3 -> 0
      '0001', # fit: 1+1=2 -> 2-7=-1 -> 0
      '0110', # fit: 6+1=7 -> 7-7=0
      '0101', # fit: 5+1=6 -> 6-7=-1 -> 0
    ):
      gi=_GI(input_len)
      gi._gen=bytearray(gi_gen.encode())
      yield gi
  gi1, gi2, gi3, gi4, gi5=(x for x in __cr_ind())
  gi_s=gi1, gi2, gi3, gi4, gi5
  _cr_ind=(x for x in (gi_s))
  cr_ind: t.Callable[[], _GI]=lambda: next(_cr_ind)
  crossover_rate=.8
  mutation_rate=.1
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema)

  for _ in range(25): # random process
    seed=rnd.random()
    rnd.seed(seed)
    exp_cgi1=rnd.choice(gi_s)
    exp_cgi2=rnd.choice(gi_s)
    rnd.seed(seed)

    # test
    ret=pop._selection()

    # results
    assert isinstance(ret, tuple)
    assert len(ret)==2
    cgi1, cgi2=ret
    assert cgi1 is exp_cgi1
    assert cgi2 is exp_cgi2

def test_selection_sum_not_zero() -> None:
  # values
  pop_num=3
  bit,min_v,max_v=4,1,11
  input_len=bit
  def calc_fitness_func(ind: _GI) -> float:
    ret=util.correct_gen_to_min_max(ind.gen, min_v, max_v)
    ret-=3
    return 0 if ret<0 else ret
  def __cr_ind() -> t.Generator[_GI]:
    for gi_gen in (
      '1000', # fit: 8+1=9 -> 9-3=6
      '0010', # fit: 2+1=3 -> 3-3=0
      '0100', # fit: 4+1=5 -> 5-3=2
    ):
      gi=_GI(input_len)
      gi._gen=bytearray(gi_gen.encode())
      yield gi
  gi1, gi2, gi3=(x for x in __cr_ind())
  _cr_ind=(x for x in (gi1, gi2, gi3))
  cr_ind: t.Callable[[], _GI]=lambda: next(_cr_ind)
  crossover_rate=.8
  mutation_rate=.1
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema)

  flag=False
  for _ in range(1000):
    # test
    ret=pop._selection()

    # results
    assert isinstance(ret, tuple)
    assert len(ret)==2
    cgi1, cgi2=ret
    flag=cgi1 is gi2 or cgi2 is gi2
    if flag:
      break

  # results
  assert flag

def test_selection_no_zero() -> None:
  # values
  pop_num=5
  bit,min_v,max_v=4,1,11
  input_len=bit
  def calc_fitness_func(ind: _GI) -> float:
    ret=util.correct_gen_to_min_max(ind.gen, min_v, max_v)
    ret-=3
    return 0 if ret<0 else ret
  def __cr_ind() -> t.Generator[_GI]:
    for gi_gen in (
      '1000', # fit: 8+1=9 -> 9-3=6
      '0011', # fit: 3+1=4 -> 4-3=1
      '0100', # fit: 4+1=5 -> 5-3=2
      '0110', # fit: 6+1=7 -> 7-3=4
      '1101', # fit: 13+1>11 -> 9+1=10 -> 10-3=7
    ):
      gi=_GI(input_len)
      gi._gen=bytearray(gi_gen.encode())
      yield gi
  gi1, gi2, gi3, gi4, gi5=(x for x in __cr_ind())
  _cr_ind=(x for x in (gi1, gi2, gi3, gi4, gi5))
  cr_ind: t.Callable[[], _GI]=lambda: next(_cr_ind)
  crossover_rate=.8
  mutation_rate=.1
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema)

  for _ in range(1000):
    # test
    ret=pop._selection()

    # results
    assert isinstance(ret, tuple)
    cgi1, cgi2=ret
    assert any(cgi1 is gi for gi in (gi1, gi2, gi3, gi4, gi5))
    assert any(cgi2 is gi for gi in (gi1, gi2, gi3, gi4, gi5))

def test_next_generation() -> None:
  # values
  pop_num=5
  bit,min_v,max_v=4,1,11
  input_len=bit
  cr_ind: t.Callable[[], _GI]=lambda: _GI(input_len)
  def calc_fitness_func(ind: _GI) -> float:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v)
  crossover_rate=.8
  mutation_rate=.1
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema)

  for _ in range(50): # random process
    # test
    pop.next_generation()

    # results
    assert len(pop.population)==pop_num
    assert len(pop.fitnesses)==pop_num

@pytest.mark.parametrize(
  'workers',
  [1, 2, 4]
)
def test_multi_vs_single_thread_consistency(workers) -> None:
  # values
  pop_num=5
  bit,min_v,max_v=4,1,11
  input_len=bit
  def __cr_ind() -> t.Generator[_GI]:
    for gi_gen in (
      '1000', # fit: 8+1=9
      '0011', # fit: 3+1=4
      '0100', # fit: 4+1=5
      '0110', # fit: 6+1=7
      '1101', # fit: 13+1>11 -> 9+1=10
    ):
      gi=_GI(input_len)
      gi._gen=bytearray(gi_gen.encode())
      yield gi
  gi1, gi2, gi3, gi4, gi5=(x for x in __cr_ind())
  _cr_ind=(x for x in (gi1, gi2, gi3, gi4, gi5))
  cr_ind: t.Callable[[], _GI]=lambda: next(_cr_ind)
  def calc_fitness_func(ind: _GI) -> float:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v)
  crossover_rate=.8
  mutation_rate=.1
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  exp_fitnesses=[9,4,5,7,10]

  # test
  pop=P(*schema, max_worker_num=workers)

  # results
  assert pop.fitnesses==exp_fitnesses

def test_error_change_population() -> None:
  # values
  pop_num=5
  bit,min_v,max_v=4,1,11
  input_len=bit
  cr_ind: t.Callable[[], _GI]=lambda: _GI(input_len)
  def calc_fitness_func(ind: _GI):
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v)
  crossover_rate=.8
  mutation_rate=.1
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema)

  # test
  with pytest.raises(AttributeError) as excinfo:
    pop.population=[cr_ind() for _ in range(pop_num)] # type: ignore

  # results
  assert str(excinfo.value)=='property \'population\' of \'Population\' object has no setter'

def test_error_change_fitnesses() -> None:
  # values
  pop_num=5
  bit,min_v,max_v=4,1,11
  input_len=bit
  cr_ind: t.Callable[[], _GI]=lambda: _GI(input_len)
  def calc_fitness_func(ind: _GI):
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v)
  crossover_rate=.8
  mutation_rate=.1
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema)

  # test
  with pytest.raises(AttributeError) as excinfo:
    pop.fitnesses=[0.]*pop_num # type: ignore

  # results
  assert str(excinfo.value)=='property \'fitnesses\' of \'Population\' object has no setter'
