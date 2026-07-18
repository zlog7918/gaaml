import json
import pytest
import typing as t
import random as rnd
from pathlib import Path
from tqdm.auto import tqdm
from gaaml.classes import _utils as util
from gaaml.classes import _consts as const
from gaaml.classes.Population import Population as P
from gaaml.classes.GenIndividual import GenIndividual as _GI
from gaaml.classes.MaxAvgMinHolder import MaxAvgMinHolder as MAMH

def bar_asserts(bar: tqdm, n: int, *, close: bool=True):
  assert bar.n==n
  assert not bar.disable
  if close:
    bar.close()
def _gen_i_path_asserts(gen_dir: Path, gi_s: list[_GI]):
  n=len(gi_s)
  assert gen_dir.is_dir()
  subdirs=tuple(gen_dir.iterdir())
  subdirs=sorted(subdirs)
  assert len(subdirs)==n
  assert {d.name for d in subdirs}=={f'ind_{i}' for i in range(n)}
  assert all(d.is_dir() for d in subdirs)
  subsubdirs=[tuple(d.iterdir()) for d in subdirs]
  subsubdirs=sorted(subsubdirs)
  model_paths=list(map(lambda dirs: next(filter(lambda dir: not dir.is_dir(), dirs)), subsubdirs))
  subsubdirs=list(map(lambda dirs, model_path: tuple(filter(lambda dir: dir!=model_path, dirs)), subsubdirs, model_paths))
  for gi, model_path in zip(gi_s, model_paths):
    assert model_path.is_file()

    with open(model_path) as model_file:
      data=json.load(model_file)

    assert data=={
      'name': _GI.__name__,
      'gen': gi.gen.decode(),
    }
  assert all(len(d)==const.NUM_OF_FIT_CALC for d in subsubdirs)
  assert all(ds.is_dir() for d in subsubdirs for ds in d)
  iter_names={f'iter_{i}' for i in range(const.NUM_OF_FIT_CALC)}
  assert all({ds.name for ds in d}==iter_names for d in subsubdirs)
def gen0_path_asserts(root_dir: Path, gi_s: tuple[_GI, ...]):
  assert root_dir.exists()
  assert root_dir.is_dir()
  gen0=tuple(root_dir.iterdir())
  assert len(gen0)==1
  gen0=gen0[0]
  assert gen0.name=='gen_0'
  _gen_i_path_asserts(gen0, list(gi_s))
def gen_i_path_asserts(root_dir: Path, gen_i: int, gi_s: list[_GI]):
  assert root_dir.exists()
  assert root_dir.is_dir()
  assert len(tuple(root_dir.iterdir()))==gen_i+1
  gen_dir=root_dir/f'gen_{gen_i}'
  assert gen_dir.exists()
  _gen_i_path_asserts(gen_dir, list(gi_s))


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

  # results
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

  # results
  assert isinstance(mgi, _GI)
  assert mgi is gi
  assert (mgi.gen==oryg_gen)!=flag

@pytest.mark.parametrize(
  ('values', 'exp_to_add'),
  [
    (([0], [1]), .01),
    (([1], [3, 5]), .01),
    (([11, 7, 6], [4], [1, 5]), .03),
    (([140], [510], [753]), 1.4),
  ]
)
def test_calc_to_add(values: tuple[list[float], ...], exp_to_add: float) -> None:
  # values
  # private access: calc_avg_from_fittnesses=P.__calc_avg_from_fittnesses
  calc_avg_from_fittnesses=t.cast(t.Callable[[list[tuple[float, float]]], float],P._Population__calc_avg_from_fittnesses) # type: ignore
  handle=MAMH[list[tuple[float, float]]](5, calc_avg_from_fittnesses)
  for v in values:
    handle.append(list(map(lambda x: (x, rnd.random()),v)))

  # test
  to_add=P._calc_to_add(handle)

  # results
  assert isinstance(float(to_add), float)
  assert to_add==pytest.approx(exp_to_add)

mark__test_create=pytest.mark.parametrize(
  ('path_to_dir_path', 'workers'),
  [
    (lambda p: str(p), 1),
    (lambda p: p, 1),
    (lambda p: str(p), 2),
    (lambda p: p, 2),
  ],
)
@mark__test_create
def test_create(
  tmp_path: Path,
  path_to_dir_path: t.Callable[[Path], Path|str],
  workers: int,
) -> None:
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
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v), rnd.random()
  crossover_rate=.8
  mutation_rate=.1
  bar=tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  exp_fitnesses=(9,4,5,7,10)
  dir_path=path_to_dir_path(tmp_path)
  dir_Path=Path(dir_path)

  # test
  pop=P(
    *schema,
    save_dir_path=dir_path,
    max_worker_num=workers,
    fitnesses_progress_output=bar,
  )

  # results
  bar_asserts(bar, pop_num)
  assert isinstance(pop.population, list)
  assert isinstance(pop.fitnesses, list)
  assert isinstance(pop.fitnesses_all, list)

  assert len(pop.population)==pop_num
  assert len(pop.fitnesses)==pop_num
  assert len(pop.fitnesses_all)==pop_num
  assert all(isinstance(ind, _GI) for ind in pop.population)
  assert all(len(ind.gen)==input_len for ind in pop.population)
  assert all(ind is gi for ind, gi in zip(pop.population, (gi1, gi2, gi3, gi4, gi5)))
  assert all(isinstance(float(fit), float) for fit in pop.fitnesses)
  assert all(isinstance(fits, list) for fits in pop.fitnesses_all)
  assert all(
    min_v<=fit
    and fit<=max_v
    and fit==exp_fit
      for fit, exp_fit in
    zip(pop.fitnesses, exp_fitnesses)
  )
  assert all(
    all(
        min_v<=fit
        and fit<=max_v
        and fit==exp_fit
      for (fit,_) in fits
    )
      for fits, exp_fit in
    zip(pop.fitnesses_all, exp_fitnesses)
  )
  gen0_path_asserts(dir_Path, (gi1, gi2, gi3, gi4, gi5))

mark__test_set_dir=pytest.mark.parametrize(
  'path_to_dir_path',
  [
    lambda p: str(p),
    lambda p: p,
  ]
)
@mark__test_set_dir
def test_set_dir(
  tmp_path: Path,
  path_to_dir_path: t.Callable[[Path], Path|str],
) -> None:
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
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v), rnd.random()
  crossover_rate=.8
  mutation_rate=.1
  bar=tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema, fitnesses_progress_output=bar)
  dir_path=path_to_dir_path(tmp_path)
  dir_Path=Path(dir_path)

  # test
  pop.set_dir(dir_path)

  # results
  bar_asserts(bar, pop_num)
  gen0_path_asserts(dir_Path, (gi1, gi2, gi3, gi4, gi5))

mark__test_set_dir_after_cr_with_path=pytest.mark.parametrize(
  ('path_to_dir_path1', 'path_to_dir_path2'),
  [
    (lambda p: str(p/'dir1'), lambda p: str(p/'dir2')),
    (lambda p: str(p/'dir1'), lambda p: (p/'dir2')),
    (lambda p: (p/'dir1'), lambda p: str(p/'dir2')),
    (lambda p: (p/'dir1'), lambda p: (p/'dir2')),
  ]
)
@mark__test_set_dir_after_cr_with_path
def test_set_dir_after_cr_with_path(
  tmp_path: Path,
  path_to_dir_path1: t.Callable[[Path], Path|str],
  path_to_dir_path2: t.Callable[[Path], Path|str],
) -> None:
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
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v), rnd.random()
  crossover_rate=.8
  mutation_rate=.1
  bar=tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  dir_path1=path_to_dir_path1(tmp_path)
  dir_path2=path_to_dir_path2(tmp_path)
  dir_Path1=Path(dir_path1)
  dir_Path2=Path(dir_path2)
  pop=P(*schema, fitnesses_progress_output=bar, save_dir_path=dir_path1)

  # test
  pop.set_dir(dir_path2)

  # results
  bar_asserts(bar, pop_num)
  assert dir_Path1.exists()
  assert len(tuple(dir_Path1.iterdir()))==0
  gen0_path_asserts(dir_Path2, (gi1, gi2, gi3, gi4, gi5))

mark__test_set_dir_after_set_dir=pytest.mark.parametrize(
  ('path_to_dir_path1', 'path_to_dir_path2'),
  [
    (lambda p: str(p/'dir1'), lambda p: str(p/'dir2')),
    (lambda p: str(p/'dir1'), lambda p: (p/'dir2')),
    (lambda p: (p/'dir1'), lambda p: str(p/'dir2')),
    (lambda p: (p/'dir1'), lambda p: (p/'dir2')),
  ]
)
@mark__test_set_dir_after_set_dir
def test_set_dir_after_set_dir(
  tmp_path: Path,
  path_to_dir_path1: t.Callable[[Path], Path|str],
  path_to_dir_path2: t.Callable[[Path], Path|str],
) -> None:
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
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v), rnd.random()
  crossover_rate=.8
  mutation_rate=.1
  bar=tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema, fitnesses_progress_output=bar)
  dir_path1=path_to_dir_path1(tmp_path)
  dir_path2=path_to_dir_path2(tmp_path)
  dir_Path1=Path(dir_path1)
  dir_Path2=Path(dir_path2)
  pop.set_dir(dir_path1)

  # test
  pop.set_dir(dir_path2)

  # results
  bar_asserts(bar, pop_num)
  assert dir_Path1.exists()
  assert len(tuple(dir_Path1.iterdir()))==0
  gen0_path_asserts(dir_Path2, (gi1, gi2, gi3, gi4, gi5))

def test_population_returns_copy() -> None:
  # values
  pop_num=2
  bit,min_v,max_v=4, 1, 11
  input_len=bit
  cr_ind: t.Callable[[], _GI]=lambda: _GI(input_len)
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v), rnd.random()
  crossover_rate=.8
  mutation_rate=.1
  bar=tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema, fitnesses_progress_output=bar)

  # test
  population_copy=pop.population
  population_copy.append(cr_ind())

  # results
  bar_asserts(bar, pop_num)
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
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v), rnd.random()
  crossover_rate=.8
  mutation_rate=.1
  bar=tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  exp_fitnesses=(9,4,5,7,10)
  pop=P(*schema, fitnesses_progress_output=bar)

  # test
  ret=pop.get_max_avg_min()

  # results
  bar_asserts(bar, pop_num)
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
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    ret=util.correct_gen_to_min_max(ind.gen, min_v, max_v)
    ret-=7
    return 0 if ret<0 else ret, rnd.random()
  def __cr_ind() -> t.Generator[_GI, None, None]:
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
  bar=tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema, fitnesses_progress_output=bar)

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
  bar_asserts(bar, pop_num)

def test_selection_sum_not_zero() -> None:
  # values
  pop_num=3
  bit,min_v,max_v=4,1,11
  input_len=bit
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    ret=util.correct_gen_to_min_max(ind.gen, min_v, max_v)
    ret-=3
    return 0 if ret<0 else ret, rnd.random()
  def __cr_ind() -> t.Generator[_GI, None, None]:
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
  bar=tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema, fitnesses_progress_output=bar)

  flag=False
  for _ in range(1000): # random procces, but possible to get gi2 even though it has fitness=0
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
  bar_asserts(bar, pop_num)
  assert flag

def test_selection_no_zero() -> None:
  # values
  pop_num=5
  bit,min_v,max_v=4,1,11
  input_len=bit
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    ret=util.correct_gen_to_min_max(ind.gen, min_v, max_v)
    ret-=3
    return 0 if ret<0 else ret, rnd.random()
  def __cr_ind() -> t.Generator[_GI, None, None]:
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
  bar=tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema, fitnesses_progress_output=bar)

  for _ in range(1000):
    # test
    ret=pop._selection()

    # results
    assert isinstance(ret, tuple)
    cgi1, cgi2=ret
    assert any(cgi1 is gi for gi in (gi1, gi2, gi3, gi4, gi5))
    assert any(cgi2 is gi for gi in (gi1, gi2, gi3, gi4, gi5))
  bar_asserts(bar, pop_num)

def test_next_generation(tmp_path: Path) -> None:
  # values
  pop_num=5
  bit,min_v,max_v=4,1,11
  input_len=bit
  cr_ind: t.Callable[[], _GI]=lambda: _GI(input_len)
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v), rnd.random()
  crossover_rate=.8
  mutation_rate=.1
  class _tqdm(tqdm):
    def __init__(self, *args, **kwargs) -> None:
      super().__init__(*args, **kwargs)
      self.reset_count=0
    def reset(self, total: float|None=None) -> None:
      self.reset_count+=1
      return super().reset(total)
  bar=_tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema, fitnesses_progress_output=bar, save_dir_path=tmp_path)

  for i in range(1, 50): # random process
    # test
    pop.next_generation(i)

    # results
    bar_asserts(bar, pop_num, close=False)
    assert bar.reset_count==i

    _pop=pop.population
    assert len(_pop)==pop_num
    assert len(pop.fitnesses)==pop_num
    gen_i_path_asserts(tmp_path, i, _pop)
  bar_asserts(bar, pop_num)

@pytest.mark.parametrize(
  'workers',
  [1, 2, 4]
)
def test_multi_vs_single_thread_consistency(workers: int) -> None:
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
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v), rnd.random()
  crossover_rate=.8
  mutation_rate=.1
  bar=tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  exp_fitnesses=[9,4,5,7,10]

  # test
  pop=P(*schema, fitnesses_progress_output=bar, max_worker_num=workers)

  # results
  bar_asserts(bar, pop_num)
  assert pop.fitnesses==exp_fitnesses

def test_error_change_population() -> None:
  # values
  pop_num=5
  bit,min_v,max_v=4,1,11
  input_len=bit
  cr_ind: t.Callable[[], _GI]=lambda: _GI(input_len)
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v), rnd.random()
  crossover_rate=.8
  mutation_rate=.1
  bar=tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema, fitnesses_progress_output=bar)

  # test
  with pytest.raises(AttributeError) as excinfo:
    pop.population=[cr_ind() for _ in range(pop_num)] # type: ignore

  # results
  bar_asserts(bar, pop_num)
  assert str(excinfo.value) in {'can\'t set attribute \'population\'', 'property \'population\' of \'Population\' object has no setter'}

def test_error_change_fitnesses() -> None:
  # values
  pop_num=5
  bit,min_v,max_v=4,1,11
  input_len=bit
  cr_ind: t.Callable[[], _GI]=lambda: _GI(input_len)
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v), rnd.random()
  crossover_rate=.8
  mutation_rate=.1
  bar=tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema, fitnesses_progress_output=bar)

  # test
  with pytest.raises(AttributeError) as excinfo:
    pop.fitnesses=[0.]*pop_num # type: ignore

  # results
  bar_asserts(bar, pop_num)
  assert str(excinfo.value) in {'can\'t set attribute \'fitnesses\'', 'property \'fitnesses\' of \'Population\' object has no setter'}

def test_error_change_fitnesses_all() -> None:
  # values
  pop_num=5
  bit,min_v,max_v=4,1,11
  input_len=bit
  cr_ind: t.Callable[[], _GI]=lambda: _GI(input_len)
  def calc_fitness_func(ind: _GI, dir: Path) -> tuple[float, float]:
    return util.correct_gen_to_min_max(ind.gen, min_v, max_v), rnd.random()
  crossover_rate=.8
  mutation_rate=.1
  bar=tqdm(total=pop_num, desc='Calculated fitnesses', position=0, mininterval=0)
  schema=pop_num, cr_ind, calc_fitness_func, crossover_rate, mutation_rate
  pop=P(*schema, fitnesses_progress_output=bar)

  # test
  with pytest.raises(AttributeError) as excinfo:
    pop.fitnesses_all=[[0.]]*pop_num # type: ignore

  # results
  bar_asserts(bar, pop_num)
  assert str(excinfo.value) in {'can\'t set attribute \'fitnesses_all\'', 'property \'fitnesses_all\' of \'Population\' object has no setter'}
