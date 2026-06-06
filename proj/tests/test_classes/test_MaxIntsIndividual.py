import json
import pytest
import typing as t
from gaaml.classes import _utils as util
from gaaml.classes.GenIndividual import GenIndividual as _GI
from gaaml.classes.MaxIntsIndividual import MaxIntsIndividual as MII

def test_create() -> None:
  # values
  x_bit=2
  x_min=1
  x_max=4
  y_bit=3
  y_min=0
  y_max=5
  schema=(
    ('x', (x_bit, x_min, x_max)),
    ('y', (y_bit, y_min, y_max)),
  )
  expected=x_bit+y_bit

  for _ in range(50): # random process
    # test
    mii=MII(schema)

    # results
    assert isinstance(mii.gen, _GI)
    assert isinstance(mii.gen.gen, bytearray)
    assert len(mii.gen.gen)==expected
    assert isinstance(mii.fenotype, dict)
    assert {*mii.fenotype.keys()}=={'x', 'y'}
    assert isinstance(mii.fenotype['x'], int)
    assert isinstance(mii.fenotype['y'], int)
    assert mii.fenotype['x']==util.correct_gen_to_min_max(mii.gen.gen[:x_bit], x_min, x_max)
    assert mii.fenotype['y']==util.correct_gen_to_min_max(mii.gen.gen[x_bit:x_bit+y_bit], y_min, y_max)

def test_mutate() -> None:
  # values
  x_bit=2
  x_min=1
  x_max=4
  y_bit=3
  y_min=0
  y_max=5
  schema=(
    ('x', (x_bit, x_min, x_max)),
    ('y', (y_bit, y_min, y_max)),
  )
  mii=MII(schema)

  for _ in range(50): # random process
    oryg_gen=mii.gen.gen[:]

    # test
    mii.mutate()

    # results
    assert mii.gen.gen!=oryg_gen
    assert isinstance(mii.fenotype, dict)
    assert {*mii.fenotype.keys()}=={'x', 'y'}
    assert isinstance(mii.fenotype['x'], int)
    assert isinstance(mii.fenotype['y'], int)
    assert mii.fenotype['x']==util.correct_gen_to_min_max(mii.gen.gen[:x_bit], x_min, x_max)
    assert mii.fenotype['y']==util.correct_gen_to_min_max(mii.gen.gen[x_bit:x_bit+y_bit], y_min, y_max)

def test_get_cp() -> None:
  # values
  x_bit=2
  x_min=1
  x_max=4
  y_bit=3
  y_min=0
  y_max=5
  schema=(
    ('x', (x_bit, x_min, x_max)),
    ('y', (y_bit, y_min, y_max)),
  )
  mii1=MII(schema)
  mii2=MII(schema)
  expected_len=x_bit+y_bit

  for _ in range(50): # random process
    # test
    cp=MII.get_cp(mii1, mii2)

    # results
    assert isinstance(cp, int)
    assert 0<=cp
    assert cp<=expected_len

mark__test_create_from_two=pytest.mark.parametrize(
  ('cp', 'expected_str'),
  [
    (0, '1 1  1 0 1'),
    (1, '1 1  1 0 1'),
    (2, '1 0  1 0 1'),
    (3, '1 0  0 0 1'),
    (4, '1 0  0 1 1'),
    (5, '1 0  0 1 0'),
  ]
)
@mark__test_create_from_two
def test_create_from_two(cp: int, expected_str: str) -> None:
  """
          x     y
  bits1: 1 0  0 1 0
  bits2: 1 1  1 0 1
  cp:3         ↑

  expected:
  1 0  0 0 1
  """
  # values
  x_bit=2
  x_min=1
  x_max=4
  y_bit=3
  y_min=0
  y_max=5
  schema=(
    ('x', (x_bit, x_min, x_max)),
    ('y', (y_bit, y_min, y_max)),
  )
  mii1=MII(schema)
  mii2=MII(schema)

  # setup
  mii1.gen._gen=bytearray('1 0  0 1 0'.replace(' ', '').encode())
  mii2.gen._gen=bytearray('1 1  1 0 1'.replace(' ', '').encode())

  # test
  mii=MII(mii1, mii2, cross_point=cp)

  # results
  assert mii.gen.gen==bytearray(expected_str.replace(' ', '').encode())
  assert {*mii.fenotype.keys()}=={'x', 'y'}
  assert isinstance(mii.fenotype['x'], int)
  assert isinstance(mii.fenotype['y'], int)
  assert mii.fenotype['x']==util.correct_gen_to_min_max(mii.gen.gen[:x_bit], x_min, x_max)
  assert mii.fenotype['y']==util.correct_gen_to_min_max(mii.gen.gen[x_bit:x_bit+y_bit], y_min, y_max)

mark__test_crossover=pytest.mark.parametrize(
  ('cp', 'expected_strs'),
  [
    (0, ('1 1  1 0 1', '1 0  0 1 0')),
    (1, ('1 1  1 0 1', '1 0  0 1 0')),
    (2, ('1 0  1 0 1', '1 1  0 1 0')),
    (3, ('1 0  0 0 1', '1 1  1 1 0')),
    (4, ('1 0  0 1 1', '1 1  1 0 0')),
    (5, ('1 0  0 1 0', '1 1  1 0 1')),
  ]
)
@mark__test_crossover
def test_crossover(cp: int, expected_strs: tuple[str, str]) -> None:
  """
          x     y
  bits1: 1 0  0 1 0
  bits2: 1 1  1 0 1
  cp:3         ↑

  expected:
  1 0  0 0 1
  1 1  1 1 0
  """
  # values
  x_bit=2
  y_bit=3
  schema=(
    ('x', (x_bit, 1, 4)),
    ('y', (y_bit, 0, 5)),
  )
  mii1=MII(schema)
  mii2=MII(schema)
  (expected_str1, expected_str2)=expected_strs

  # setup
  mii1.gen._gen=bytearray('1 0  0 1 0'.replace(' ', '').encode())
  mii2.gen._gen=bytearray('1 1  1 0 1'.replace(' ', '').encode())

  # test
  li_s=MII.crossover(mii1, mii2, cp)

  # results
  assert isinstance(li_s, tuple)
  assert li_s[0].gen.gen==bytearray(expected_str1.replace(' ', '').encode())
  assert li_s[1].gen.gen==bytearray(expected_str2.replace(' ', '').encode())

mark__test_update_fenotype=pytest.mark.parametrize(
  ('b_str', 'expected'),
  [
    ('1 0  1 1 0', {'x': 3, 'y': 4}),
    ('0 0  1 1 0', {'x': 1, 'y': 4}),
    ('0 1  1 1 0', {'x': 2, 'y': 4}),
    ('1 1  1 1 0', {'x': 4, 'y': 4}),
    ('1 0  1 1 1', {'x': 3, 'y': 5}),
    ('1 0  0 0 0', {'x': 3, 'y': 0}),
    ('1 0  0 0 1', {'x': 3, 'y': 1}),
    ('1 0  0 1 0', {'x': 3, 'y': 2}),
    ('1 0  0 1 1', {'x': 3, 'y': 3}),
    ('1 0  1 0 0', {'x': 3, 'y': 4}),
    ('1 0  1 0 1', {'x': 3, 'y': 5}),
  ]
)
@mark__test_update_fenotype
def test_update_fenotype(b_str: str, expected: dict[str, int]) -> None:
  """
  x: 2, 1, 4
  x: 3, 0, 5
         x     y
  bits: 1 0  1 1 0

  expected:
  x=2, y=6
  fenotype: {
    x: 2+1=3
    y: 6+0>5 -> 6->4 -> 4+0=4
  }
  """
  # values
  x_bit=2
  y_bit=3
  schema=(
    ('x', (x_bit, 1, 4)),
    ('y', (y_bit, 0, 5)),
  )
  mii=MII(schema)

  # setup
  mii.gen._gen=bytearray('0 0  0 0 0'.replace(' ', '').encode())
  mii._update_fenotype()
  oryg_fenotype1={k: v for k,v in mii.fenotype.items()}
  mii.gen._gen=bytearray(b_str.replace(' ', '').encode())
  oryg_fenotype2={k: v for k,v in mii.fenotype.items()}

  # test
  mii._update_fenotype()

  # results
  assert oryg_fenotype1==oryg_fenotype2
  assert oryg_fenotype2!=mii.fenotype
  assert mii.fenotype==expected

mark__test_get_fenotype=pytest.mark.parametrize(
  ('b_str', 'expected'),
  [
    ('1 0  1 1 0', {'x': 3, 'y': 4}),
    ('0 0  1 1 0', {'x': 1, 'y': 4}),
    ('0 1  1 1 0', {'x': 2, 'y': 4}),
    ('1 1  1 1 0', {'x': 4, 'y': 4}),
    ('1 0  1 1 1', {'x': 3, 'y': 5}),
    ('1 0  0 0 0', {'x': 3, 'y': 0}),
    ('1 0  0 0 1', {'x': 3, 'y': 1}),
    ('1 0  0 1 0', {'x': 3, 'y': 2}),
    ('1 0  0 1 1', {'x': 3, 'y': 3}),
    ('1 0  1 0 0', {'x': 3, 'y': 4}),
    ('1 0  1 0 1', {'x': 3, 'y': 5}),
  ]
)
@mark__test_get_fenotype
def test_get_fenotype(b_str: str, expected: dict[str, int]) -> None:
  """
  x: 2, 1, 4
  x: 3, 0, 5
         x     y
  bits: 1 0  1 1 0

  expected:
  x=2, y=6
  fenotype: {
    x: 2+1=3
    y: 6+0>5 -> 6->4 -> 4+0=4
  }
  """
  # values
  x_bit=2
  y_bit=3
  schema=(
    ('x', (x_bit, 1, 4)),
    ('y', (y_bit, 0, 5)),
  )
  mii=MII(schema)
  # private access: mii.__schema
  _schema=mii._MaxIntsIndividual__schema # type: ignore

  # setup
  mii.gen._gen=bytearray(b_str.replace(' ', '').encode())

  # test
  fenotype=MII.get_fenotype(mii.gen, _schema)

  # results
  assert fenotype==expected

def test_save_format() -> None:
  # values
  schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  mii=MII(schema)

  # test
  result=mii._save_format()

  # results
  assert isinstance(result, dict)
  assert result=={
    'name': MII.__name__,
    'gen': mii._gen._save_format(),
    'schema': [
      ['x', 2, 1, 4],
      ['y', 3, 0, 5],
    ],
  }

def test_save_format_returns_serializable_data():
  # values
  schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  mii=MII(schema)
  result=mii._save_format()

  # test/results
  _=json.dumps(result)

def test_load_from_format() -> None:
  # values
  schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  input_len=sum(l for _, (l, _, _) in schema)
  gi=_GI(input_len)
  saved_model={
    'name': MII.__name__,
    'gen': gi._save_format(),
    'schema': [
      ['x', 2, 1, 4],
      ['y', 3, 0, 5],
    ],
  }

  # test
  loaded=MII._load_from_format(saved_model)

  # results
  assert isinstance(loaded, MII)
  assert loaded._gen._gen==gi._gen
  # private access: loaded.__schema
  loaded_schema=loaded._MaxIntsIndividual__schema # type: ignore
  assert loaded_schema==schema

def test_load_from_format_roundtrip() -> None:
  # values
  schema=(
    ('x', (2, 1, 4)),
    ('y', (3, 0, 5)),
  )
  mii=MII(schema)

  # setup
  saved=json.loads(json.dumps(mii._save_format()))

  # test
  loaded=MII._load_from_format(saved)

  # results
  assert isinstance(loaded, MII)
  assert loaded._gen._gen==mii._gen._gen

def test_error_name_collition_on_create() -> None:
  # values
  schema=(
    ('x', (2, 1, 4)),
    ('x', (6, 1, 12)),
    ('y', (3, 0, 5))
  )

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MII(schema)

  # results
  assert str(excinfo.value)=='Names can not collide'

mark__test_error_not_same_conf_on_get_cp=pytest.mark.parametrize(
  'schema2',
  [
    (('x', (2, 2, 4)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (3, 1, 5))),
    (('x', (2, 1, 3)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (3, 0, 4))),
    (('x', (3, 1, 4)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (4, 0, 5))),
  ]
)
@mark__test_error_not_same_conf_on_get_cp
def test_error_not_same_conf_on_get_cp(schema2: tuple[tuple[str, tuple[int, int, int]], tuple[str, tuple[int, int, int]]]) -> None:
  # values
  schema1=(('x', (2, 1, 4)), ('y', (3, 0, 5)))
  mii1=MII(schema1)
  mii2=MII(schema2)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MII.get_cp(mii1, mii2)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal configuration'

mark__test_error_not_same_conf_on_crossover=pytest.mark.parametrize(
  'schema2',
  [
    (('x', (2, 2, 4)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (3, 1, 5))),
    (('x', (2, 1, 3)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (3, 0, 4))),
    (('x', (3, 1, 4)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (4, 0, 5))),
  ]
)
@mark__test_error_not_same_conf_on_crossover
def test_error_not_same_conf_on_crossover(schema2: tuple[tuple[str, tuple[int, int, int]], tuple[str, tuple[int, int, int]]]) -> None:
  # values
  schema1=(('x', (2, 1, 4)), ('y', (3, 0, 5)))
  mii1=MII(schema1)
  mii2=MII(schema2)
  dummy_cp=1

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MII.crossover(mii1, mii2, dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal configuration'

mark__test_error_not_same_conf_on_create=pytest.mark.parametrize(
  'schema2',
  [
    (('x', (2, 2, 4)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (3, 1, 5))),
    (('x', (2, 1, 3)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (3, 0, 4))),
    (('x', (3, 1, 4)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (4, 0, 5))),
  ]
)
@mark__test_error_not_same_conf_on_create
def test_error_not_same_conf_on_create(schema2: tuple[tuple[str, tuple[int, int, int]], tuple[str, tuple[int, int, int]]]) -> None:
  # values
  schema1=(('x', (2, 1, 4)), ('y', (3, 0, 5)))
  mii1=MII(schema1)
  mii2=MII(schema2)
  dummy_cp=1

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MII(mii1, mii2, cross_point=dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal configuration'

mark__test_error_outside_range_on_create=pytest.mark.parametrize(
  'cp',
  [
    -1,
    6,
  ]
)
@mark__test_error_outside_range_on_create
def test_error_outside_range_on_create(cp: int) -> None:
  # values
  x_bit=2
  y_bit=3
  schema=(
    ('x', (x_bit, 1, 4)),
    ('y', (y_bit, 0, 5)),
  )
  mii1=MII(schema)
  mii2=MII(schema)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MII(mii1, mii2, cross_point=cp)

  # results
  assert str(excinfo.value)=='Cross point is outside of solution'

mark__test_error_outside_range_on_crossover=pytest.mark.parametrize(
  'cp',
  [
    -1,
    6,
  ]
)
@mark__test_error_outside_range_on_crossover
def test_error_outside_range_on_crossover(cp: int) -> None:
  # values
  x_bit=2
  y_bit=3
  schema=(
    ('x', (x_bit, 1, 4)),
    ('y', (y_bit, 0, 5)),
  )
  mii1=MII(schema)
  mii2=MII(schema)
  expected_len=x_bit+y_bit

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MII.crossover(mii1, mii2, cp)

  # results
  assert str(excinfo.value)=='Cross point is outside of solution'

mark__test_error_load_invalid_model=pytest.mark.parametrize(
  'saved',
  [
    # invalid keys
    {},
    {
      'name': MII.__name__,
    },
    {
      'gen': _GI(5)._save_format(),
    },
    {
      'name': MII.__name__,
      'gen': _GI(5)._save_format(),
    },
    {
      'name': MII.__name__,
      'gen': _GI(5)._save_format(),
      'schema': [['x', 2, 1, 4]],
      'extra': 123,
    },

    # wrong name
    {
      'name': 'OtherClass',
      'gen': _GI(5)._save_format(),
      'schema': [['x', 2, 1, 4]],
    },

    # gen wrong type
    {
      'name': MII.__name__,
      'gen': [],
      'schema': [['x', 2, 1, 4]],
    },

    # schema wrong type
    {
      'name': MII.__name__,
      'gen': _GI(5)._save_format(),
      'schema': {},
    },

    # schema entry too short
    {
      'name': MII.__name__,
      'gen': _GI(5)._save_format(),
      'schema': [['x', 2, 1]],
    },

    # schema entry too long
    {
      'name': MII.__name__,
      'gen': _GI(5)._save_format(),
      'schema': [['x', 2, 1, 4, 5]],
    },

    # schema entry wrong types
    {
      'name': MII.__name__,
      'gen': _GI(5)._save_format(),
      'schema': [['x', '2', 1, 4]],
    },
    {
      'name': MII.__name__,
      'gen': _GI(5)._save_format(),
      'schema': [[1, 2, 1, 4]],
    },

    # schema entry not list
    {
      'name': MII.__name__,
      'gen': _GI(5)._save_format(),
      'schema': ['not-an-entry'],
    },

    # duplicate names
    {
      'name': MII.__name__,
      'gen': _GI(5)._save_format(),
      'schema': [
        ['x', 2, 1, 4],
        ['x', 3, 0, 5],
      ],
    },

    # invalid nested GenIndividual model
    {
      'name': MII.__name__,
      'gen': {'invalid': True},
      'schema': [['x', 2, 1, 4]],
    },
  ]
)
@mark__test_error_load_invalid_model
def test_error_load_from_format_invalid_model(saved: dict[str, object]) -> None:
  # test
  with pytest.raises(ValueError) as excinfo:
    _=MII._load_from_format(saved)

  # results
  assert str(excinfo.value)==f'Model saved is not {MII.__name__}'

mark__test_error_illegal_argument_on_create=pytest.mark.parametrize(
  'func_args_kwargs',
  [
    lambda mii1, mii2: ((mii1, None), {'cross_point': 1}),
    lambda mii1, mii2: ((mii1, mii2), {}),
    lambda mii1, mii2: ((mii1, mii2), {'cross_point': None}),
    lambda mii1, mii2: ((mii1, None), {}),
    lambda mii1, mii2: ((mii1, None), {'cross_point': None}),
  ],
)
@mark__test_error_illegal_argument_on_create
def test_error_illegal_argument_on_create(
  func_args_kwargs: t.Callable[[MII, MII], tuple[tuple[t.Any, ...], dict[str, t.Any]]],
) -> None:
  # values
  x_bit=2
  y_bit=3
  schema=(
    ('x', (x_bit, 1, 4)),
    ('y', (y_bit, 0, 5)),
  )
  mii1=MII(schema)
  mii2=MII(schema)
  args, kwargs=func_args_kwargs(mii1, mii2)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MII(*args, **kwargs)

  # results
  assert str(excinfo.value)=='Illegal argument options'
