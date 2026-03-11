import pytest
from gaaml.classes import _utils as util
from gaaml.classes.GenIndividual import GenIndividual as GI
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

  for i in range(1, 50): # random process
    # test
    mii=MII(schema)

    # results
    assert isinstance(mii.gen, GI)
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

  for i in range(1, 50): # random process
    mii=MII(schema)
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

  for i in range(1, 50): # random process
    # test
    cp=MII.get_cp(mii1, mii2)

    # results
    assert isinstance(cp, int)
    assert 0<=cp
    assert cp<=expected_len

def test_create_from_two() -> None:
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
  expected_len=x_bit+y_bit
  mii1.gen.gen=bytearray('1 0  0 1 0'.replace(' ', '').encode())
  mii2.gen.gen=bytearray('1 1  1 0 1'.replace(' ', '').encode())

  for cp, expected_str in zip(range(expected_len+1), (
    '1 1  1 0 1',
    '1 1  1 0 1',
    '1 0  1 0 1',
    '1 0  0 0 1',
    '1 0  0 1 1',
    '1 0  0 1 0',
  )):
    # test
    mii=MII(mii1, mii2, cross_point=cp)

    # results
    assert mii.gen.gen==bytearray(expected_str.replace(' ', '').encode())
    assert {*mii.fenotype.keys()}=={'x', 'y'}
    assert isinstance(mii.fenotype['x'], int)
    assert isinstance(mii.fenotype['y'], int)
    assert mii.fenotype['x']==util.correct_gen_to_min_max(mii.gen.gen[:x_bit], x_min, x_max)
    assert mii.fenotype['y']==util.correct_gen_to_min_max(mii.gen.gen[x_bit:x_bit+y_bit], y_min, y_max)

def test_crossover() -> None:
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
  expected_len=x_bit+y_bit
  mii1.gen.gen=bytearray('1 0  0 1 0'.replace(' ', '').encode())
  mii2.gen.gen=bytearray('1 1  1 0 1'.replace(' ', '').encode())

  for cp, (expected_str1, expected_str2) in zip(range(expected_len+1), (
    ('1 1  1 0 1', '1 0  0 1 0'),
    ('1 1  1 0 1', '1 0  0 1 0'),
    ('1 0  1 0 1', '1 1  0 1 0'),
    ('1 0  0 0 1', '1 1  1 1 0'),
    ('1 0  0 1 1', '1 1  1 0 0'),
    ('1 0  0 1 0', '1 1  1 0 1'),
  )):
    # test
    li_s=MII.crossover(mii1, mii2, cp)

    # results
    assert isinstance(li_s, tuple)
    assert li_s[0].gen.gen==bytearray(expected_str1.replace(' ', '').encode())
    assert li_s[1].gen.gen==bytearray(expected_str2.replace(' ', '').encode())

def test_update_fenotype() -> None:
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
  mii.gen.gen=bytearray('0 0  0 0 0'.replace(' ', '').encode())
  mii._update_fenotype()

  for b_str, expected in (
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
  ):
    oryg_fenotype1={k: v for k,v in mii.fenotype.items()}
    mii.gen.gen=bytearray(b_str.replace(' ', '').encode())
    oryg_fenotype2={k: v for k,v in mii.fenotype.items()}

    # test
    mii._update_fenotype()

    # results
    assert oryg_fenotype1==oryg_fenotype2
    assert oryg_fenotype2!=mii.fenotype
    assert mii.fenotype==expected

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

def test_error_not_same_conf_on_get_cp() -> None:
  # values
  schema1=(('x', (2, 1, 4)), ('y', (3, 0, 5)))
  mii1=MII(schema1)
  for schema2 in (
    (('x', (2, 2, 4)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (3, 1, 5))),
    (('x', (2, 1, 3)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (3, 0, 4))),
    (('x', (3, 1, 4)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (4, 0, 5))),
  ):
    mii2=MII(schema2)

    # test
    with pytest.raises(ValueError) as excinfo:
      _=MII.get_cp(mii1, mii2)

    # results
    assert str(excinfo.value)=='First and second solution do not have equal configuration'

def test_error_not_same_conf_on_crossover() -> None:
  # values
  schema1=(('x', (2, 1, 4)), ('y', (3, 0, 5)))
  mii1=MII(schema1)
  dummy_cp=1
  for schema2 in (
    (('x', (2, 2, 4)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (3, 1, 5))),
    (('x', (2, 1, 3)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (3, 0, 4))),
    (('x', (3, 1, 4)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (4, 0, 5))),
  ):
    mii2=MII(schema2)

    # test
    with pytest.raises(ValueError) as excinfo:
      _=MII.crossover(mii1, mii2, dummy_cp)

    # results
    assert str(excinfo.value)=='First and second solution do not have equal configuration'

def test_error_not_same_conf_on_create() -> None:
  # values
  schema1=(('x', (2, 1, 4)), ('y', (3, 0, 5)))
  mii1=MII(schema1)
  dummy_cp=1
  for schema2 in (
    (('x', (2, 2, 4)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (3, 1, 5))),
    (('x', (2, 1, 3)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (3, 0, 4))),
    (('x', (3, 1, 4)), ('y', (3, 0, 5))),
    (('x', (2, 1, 4)), ('y', (4, 0, 5))),
  ):
    mii2=MII(schema2)

    # test
    with pytest.raises(ValueError) as excinfo:
      _=MII(mii1, mii2, cross_point=dummy_cp)

    # results
    assert str(excinfo.value)=='First and second solution do not have equal configuration'

def test_error_outside_range_on_create() -> None:
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

  for cp in (
    -1,
    expected_len+1,
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=MII(mii1, mii2, cross_point=cp)

    # results
    assert str(excinfo.value)=='Cross point is outside of solution'

def test_error_outside_range_on_crossover() -> None:
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

  for cp in (
    -1,
    expected_len+1,
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=MII.crossover(mii1, mii2, cp)

    # results
    assert str(excinfo.value)=='Cross point is outside of solution'

def test_error_illegal_argument_on_create() -> None:
  # values
  x_bit=2
  y_bit=3
  schema=(
    ('x', (x_bit, 1, 4)),
    ('y', (y_bit, 0, 5)),
  )
  mii1=MII(schema)
  mii2=MII(schema)

  for args, kwargs in (
    ((mii1, None), {'cross_point': 1}),
    ((mii1, mii2), {}),
    ((mii1, mii2), {'cross_point': None}),
    ((mii1, None), {}),
    ((mii1, None), {'cross_point': None}),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=MII(*args, **kwargs) # type: ignore

    # results
    assert str(excinfo.value)=='Illegal argument options'
