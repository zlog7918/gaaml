import json
import pytest
import typing as t
from gaaml.classes.GenIndividual import GenIndividual as GI

def test_create() -> None:
  # values
  input_len=5

  for _ in range(50): # random process
    # test
    gi=GI(input_len)

    # results
    assert isinstance(gi.gen, bytearray)
    assert len(gi.gen)==input_len

def test_load_from_format() -> None:
  # values
  gen_str='10110'

  # test
  gi=GI._load_from_format({
    'name': GI.__name__,
    'gen': gen_str,
  })

  # results
  assert isinstance(gi, GI)
  assert gi.gen==bytearray(gen_str.encode())

def test_load_from_format_roundtrip() -> None:
  # values
  input_len=5
  gi1=GI(input_len)

  # setup
  gi1._gen=bytearray(b'10110')

  # test
  saved=gi1._save_format()
  gi2=GI._load_from_format(saved)

  # results
  assert isinstance(gi2, GI)
  assert gi2 is not gi1
  assert gi2.gen==gi1.gen
  assert gi2._gen==gi1._gen

def test_mutate() -> None:
  # values
  input_len=5

  for _ in range(50): # random process
    gi=GI(input_len)
    oryg_gen=gi.gen[:]

    # test
    gi.mutate()

    # results
    assert gi.gen!=oryg_gen

def test_get_cp() -> None:
  # values
  input_len=5
  gi1=GI(input_len)
  gi2=GI(input_len)
  for _ in range(50): # random process
    # test
    cp=GI.get_cp(gi1, gi2)

    # results
    assert isinstance(cp, int)
    assert 0<=cp
    assert cp<=input_len

mark__test_create_from_two=pytest.mark.parametrize(
  ('cp', 'expected_str'),
  [
    (0, '10001'),
    (1, '00001'),
    (2, '01001'),
    (3, '01101'),
    (4, '01101'),
    (5, '01100'),
  ],
)
@mark__test_create_from_two
def test_create_from_two(cp: int, expected_str: str) -> None:
  """
  0 1 1|0 0
  1 0 0|0 1
  cp:3 ↑

  expected:
  0 1 1|0 1
  """
  # values
  input_len=5
  gi1=GI(input_len)
  gi2=GI(input_len)

  # setup
  gi1._gen=bytearray('01100'.encode())
  gi2._gen=bytearray('10001'.encode())

  # test
  gi=GI(gi1, gi2, cross_point=cp)

  # results
  assert gi.gen==bytearray(expected_str.encode())
  assert len(gi.gen)==input_len

mark__test_crossover=pytest.mark.parametrize(
  ('cp', 'expected_strs'),
  [
    (0, ('10001', '01100')),
    (1, ('00001', '11100')),
    (2, ('01001', '10100')),
    (3, ('01101', '10000')),
    (4, ('01101', '10000')),
    (5, ('01100', '10001')),
  ],
)
@mark__test_crossover
def test_crossover(cp: int, expected_strs: tuple[str, str]) -> None:
  """
  0 1 1|0 0
  1 0 0|0 1
  cp:3 ↑

  expected:
  0 1 1|0 1
  1 0 0|0 0
  """
  # values
  input_len=5
  gi1=GI(input_len)
  gi2=GI(input_len)

  # setup
  gi1._gen=bytearray('01100'.encode())
  gi2._gen=bytearray('10001'.encode())

  # test
  gi_s=GI.crossover(gi1, gi2, cp)

  # results
  assert isinstance(gi_s, tuple)
  assert isinstance(gi_s[0], GI)
  assert gi_s[0].gen==bytearray(expected_strs[0].encode())
  assert isinstance(gi_s[1], GI)
  assert gi_s[1].gen==bytearray(expected_strs[1].encode())

def test_save_format() -> None:
  # values
  input_len=5
  gen_str='1 0 1 1 0'
  gi=GI(input_len)

  # setup
  gi._gen=bytearray(gen_str.replace(' ', '').encode())

  # test
  result=gi._save_format()

  # results
  assert isinstance(result, dict)
  assert result=={
    'name': GI.__name__,
    'gen': gen_str.replace(' ', ''),
  }

def test_save_format_returns_serializable_data():
  # values
  input_len=5
  gi=GI(input_len)
  result=gi._save_format()

  # test/results
  _=json.dumps(result)

mark__test_error_not_same_type=pytest.mark.parametrize(
  'func',
  [
    lambda gi1, gi2: GI.get_cp(gi1, gi2),
    lambda gi1, gi2: GI.crossover(gi1, gi2, 2),
    lambda gi1, gi2: GI(gi1, gi2, cross_point=2),
  ],
)
@mark__test_error_not_same_type
def test_error_not_same_type(
  func: t.Callable[[GI, GI], object],
) -> None:
  # values
  input_len1=5
  input_len2=6
  gi1=GI(input_len1)
  gi2=GI(input_len2)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=func(gi1, gi2)

  # results
  assert str(excinfo.value)=='First and second solution are not equal in size'


mark__test_error_outside_range=pytest.mark.parametrize(
  ('func', 'cp'),
  [
    (lambda gi1, gi2, cp: GI.crossover(gi1, gi2, cp), -1),
    (lambda gi1, gi2, cp: GI.crossover(gi1, gi2, cp), 6),
    (lambda gi1, gi2, cp: GI(gi1, gi2, cross_point=cp), -1),
    (lambda gi1, gi2, cp: GI(gi1, gi2, cross_point=cp), 6),
  ],
)
@mark__test_error_outside_range
def test_error_outside_range(
  func: t.Callable[[GI, GI, int], object],
  cp: int,
) -> None:
  # values
  input_len=5
  gi1=GI(input_len)
  gi2=GI(input_len)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=func(gi1, gi2, cp)

  # results
  assert str(excinfo.value)=='Cross point is outside of solution'

mark__test_error_invalid_saved_model=pytest.mark.parametrize(
  'saved_model',
  [
    # invalid keys
    {'name': GI.__name__},
    {'gen': '10110'},
    {'name': GI.__name__, 'gen': '10110', 'extra': 1},

    # invalid name
    {'name': 'OtherClass', 'gen': '10110'},
    {'name': '_GI', 'gen': '10110'},
    {'name': 'GI', 'gen': '10110'},

    # invalid gen type
    {'name': GI.__name__, 'gen': 10110},
    {'name': GI.__name__, 'gen': None},
    {'name': GI.__name__, 'gen': b'10110'},

    # invalid gen contents
    {'name': GI.__name__, 'gen': ''},
    {'name': GI.__name__, 'gen': '10112'},
    {'name': GI.__name__, 'gen': '10a10'},
    {'name': GI.__name__, 'gen': '10 10'},
  ],
)
@mark__test_error_invalid_saved_model
def test_error_invalid_saved_model(
  saved_model: dict[str, t.Any],
) -> None:
  # values ^

  # test
  with pytest.raises(ValueError) as excinfo:
    _=GI._load_from_format(saved_model)

  # results
  assert str(excinfo.value)==f'Model saved is not {GI.__name__}'

mark__test_error_illegal_argument_on_create=pytest.mark.parametrize(
  'func_args_kwargs',
  [
    lambda gi1, gi2: ((gi1, None), {'cross_point': 2}),
    lambda gi1, gi2: ((gi1, gi2), {}),
    lambda gi1, gi2: ((gi1, gi2), {'cross_point': None}),
    lambda gi1, gi2: ((gi1, None), {}),
    lambda gi1, gi2: ((gi1, None), {'cross_point': None}),
  ],
)
@mark__test_error_illegal_argument_on_create
def test_error_illegal_argument_on_create(
  func_args_kwargs: t.Callable[[GI, GI], tuple[tuple[t.Any, ...], dict[str, t.Any]]],
) -> None:
  # values
  input_len=5
  gi1=GI(input_len)
  gi2=GI(input_len)
  args, kwargs=func_args_kwargs(gi1, gi2)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=GI(*args, **kwargs)

  # results
  assert str(excinfo.value)=='Illegal argument options'
