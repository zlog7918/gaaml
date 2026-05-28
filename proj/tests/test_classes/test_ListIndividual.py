import pytest
import typing as t
from gaaml.classes.ListIndividual import ListIndividual as LI

def test_create() -> None:
  # values
  input_len=5
  item_size=3
  min_list_size=1
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)

  for _ in range(50): # random process
    # test
    li=LI(input_len, schema)

    # results
    assert isinstance(li.gen, bytearray)
    assert len(li.gen)==input_len*item_size

def test_save_format() -> None:
  # values
  input_len=5
  item_size=3
  min_list_size=1
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  gen_str='0 1 1  1 0 0  1 1 1  0 0 1  1 0 1'
  li=LI(input_len, schema)

  # setup
  li._gen=bytearray(gen_str.replace(' ', '').encode())

  # test
  result=li._save_format()

  # results
  assert isinstance(result, dict)
  assert result=={
    'name': LI.__name__,
    'gen': gen_str.replace(' ', ''),
  }

def test_save_format_empty_gen() -> None:
  # values
  input_len=0
  item_size=1
  min_list_size=0
  max_list_size=10
  schema=((min_list_size, max_list_size), item_size)

  li=LI(input_len, schema)

  # test
  result=li._save_format()

  # results
  assert isinstance(result, dict)
  assert result=={
    'name': LI.__name__,
    'gen': '',
  }

def test_mutate() -> None:
  # values
  input_len=5
  item_size=3
  min_list_size=1
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)

  for _ in range(50): # random process
    li=LI(input_len, schema)
    oryg_gen=li.gen[:]

    # test
    li.mutate()

    # results
    assert li.gen!=oryg_gen

def test_get_cp() -> None:
  # values
  input_len1=5
  input_len2=20
  item_size=3
  min_list_size=1
  max_list_size=50
  bit_len1=input_len1*item_size
  bit_len2=input_len2*item_size
  min_bit_len=min_list_size*item_size
  max_bit_len=max_list_size*item_size
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema)
  li2=LI(input_len2, schema)

  for _ in range(50): # random process
    # test
    cp=LI.get_cp(li1, li2)

    # results
    assert isinstance(cp, tuple)
    cp1, cp2=cp
    assert 0<=cp1
    assert 0<=cp2
    assert cp1<=bit_len1
    assert cp2<=bit_len2
    assert cp1%item_size==cp2%item_size
    assert cp1+(bit_len2-cp2)>=min_bit_len
    assert (bit_len1-cp1)+cp2>=min_bit_len

mark__test_create_from_two=pytest.mark.parametrize(
  ('cp', 'expected_str'),
  [
    ((4, 7), '0 1 1  1 0 1'),
    ((4, 4), '0 1 1  1 1 0  0 0 1'),
    ((4, 1), '0 1 1  1 1 0  0 1 0  0 0 1'),
    ((1, 7), '0 0 1'),
    ((6, 0), '0 1 1  1 1 1  1 1 0  0 1 0  0 0 1'),
    ((6, 3), '0 1 1  1 1 1  0 1 0  0 0 1'),
    ((6, 9), '0 1 1  1 1 1'),
  ],
)
@mark__test_create_from_two
def test_create_from_two(cp: tuple[int, int], expected_str: str) -> None:
  """
  0 1 1  1 1 1
          ↑
  1 1 0  0 1 0  0 0 1
  cp:4,7         ↑

  expected:
  0 1 1  1 0 1
  """
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=1
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema)
  li2=LI(input_len2, schema)

  # setup
  li1._gen=bytearray('0 1 1  1 1 1'.replace(' ', '').encode())
  li2._gen=bytearray('1 1 0  0 1 0  0 0 1'.replace(' ', '').encode())

  # test
  li=LI(li1, li2, cross_point=cp)

  # results
  assert isinstance(li, LI)
  assert li.gen==bytearray(expected_str.replace(' ', '').encode())

mark__test_create_from_two_too_long=pytest.mark.parametrize(
  ('cp', 'expected_str'),
  [
    ((4, 1), '0 1 1  1 1 0  0 1 0'), # '0 1 1  1 1 0  0 1 0  0 0 1'
    ((6, 0), '0 1 1  1 1 1  1 1 0'), # '0 1 1  1 1 1  1 1 0  0 1 0  0 0 1'
    ((6, 3), '0 1 1  1 1 1  0 1 0'), # '0 1 1  1 1 1  0 1 0  0 0 1'
  ],
)
@mark__test_create_from_two_too_long
def test_create_from_two_too_long(cp: tuple[int, int], expected_str: str) -> None:
  """
  0 1 1  1 1 1
          ↑
  1 1 0  0 1 0  0 0 1
  cp:4,7         ↑

  expected:
  0 1 1  1 0 1
  """
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=1
  max_list_size=3
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema)
  li2=LI(input_len2, schema)

  # setup
  li1._gen=bytearray('0 1 1  1 1 1'.replace(' ', '').encode())
  li2._gen=bytearray('1 1 0  0 1 0  0 0 1'.replace(' ', '').encode())

  # test
  li=LI(li1, li2, cross_point=cp)

  # results
  assert isinstance(li, LI)
  assert li.gen==bytearray(expected_str.replace(' ', '').encode())

mark__test_crossover=pytest.mark.parametrize(
  ('cp', 'expected_strs'),
  [
    ((4, 7), ('0 1 1  1 0 1', '1 1 0  0 1 0  0 1 1')),
    ((4, 4), ('0 1 1  1 1 0  0 0 1', '1 1 0  0 1 1')),
    ((4, 1), ('0 1 1  1 1 0  0 1 0  0 0 1', '1 1 1')),
    ((1, 7), ('0 0 1', '1 1 0  0 1 0  0 1 1  1 1 1')),
    # ((6, 0), ('0 1 1  1 1 1  1 1 0  0 1 0  0 0 1', '')), # throws error
    ((6, 3), ('0 1 1  1 1 1  0 1 0  0 0 1', '1 1 0')),
    ((6, 9), ('0 1 1  1 1 1', '1 1 0  0 1 0  0 0 1')),
  ],
)
@mark__test_crossover
def test_crossover(cp: tuple[int, int], expected_strs: tuple[str, str]) -> None:
  """
  0 1 1  1 1 1
          ↑
  1 1 0  0 1 0  0 0 1
  cp:4,7         ↑

  expected:
  0 1 1  1 0 1
  1 1 0  0 1 0  0 1 1
  """
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=1
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema)
  li2=LI(input_len2, schema)

  # setup
  li1._gen=bytearray('0 1 1  1 1 1'.replace(' ', '').encode())
  li2._gen=bytearray('1 1 0  0 1 0  0 0 1'.replace(' ', '').encode())
  (expected_str1, expected_str2)=expected_strs

  # test
  li_s=LI.crossover(li1, li2, cp)

  # results
  assert isinstance(li_s, tuple)
  assert li_s[0].gen==bytearray(expected_str1.replace(' ', '').encode())
  assert li_s[1].gen==bytearray(expected_str2.replace(' ', '').encode())

mark__test_crossover_too_long=pytest.mark.parametrize(
  ('cp', 'expected_strs'),
  [
    ((4, 1), ('0 1 1  1 1 0  0 1 0', '1 1 1')), # ('0 1 1  1 1 0  0 1 0  0 0 1', '1 1 1')
    ((1, 7), ('0 0 1', '1 1 0  0 1 0  0 1 1')), # ('0 0 1', '1 1 0  0 1 0  0 1 1  1 1 1')
    ((6, 3), ('0 1 1  1 1 1  0 1 0', '1 1 0')), # ('0 1 1  1 1 1  0 1 0  0 0 1', '1 1 0')
  ],
)
@mark__test_crossover_too_long
def test_crossover_too_long(cp: tuple[int, int], expected_strs: tuple[str, str]) -> None:
  """
  0 1 1  1 1 1
          ↑
  1 1 0  0 1 0  0 0 1
  cp:4,7         ↑

  expected:
  0 1 1  1 0 1
  1 1 0  0 1 0  0 1 1
  """
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=1
  max_list_size=3
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema)
  li2=LI(input_len2, schema)

  # setup
  li1._gen=bytearray('0 1 1  1 1 1'.replace(' ', '').encode())
  li2._gen=bytearray('1 1 0  0 1 0  0 0 1'.replace(' ', '').encode())
  (expected_str1, expected_str2)=expected_strs

  # test
  li_s=LI.crossover(li1, li2, cp)

  # results
  assert isinstance(li_s, tuple)
  assert li_s[0].gen==bytearray(expected_str1.replace(' ', '').encode())
  assert li_s[1].gen==bytearray(expected_str2.replace(' ', '').encode())

def test_error_list_size_too_small_on_create() -> None:
  # values
  input_len=2
  item_size=3
  min_list_size=3
  max_list_size=7
  schema=((min_list_size, max_list_size), item_size)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI(input_len, schema)

  # results
  assert str(excinfo.value)=='List size out of allowed range'

def test_error_list_size_too_big_on_create() -> None:
  # values
  input_len=9
  item_size=3
  min_list_size=3
  max_list_size=7
  schema=((min_list_size, max_list_size), item_size)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI(input_len, schema)

  # results
  assert str(excinfo.value)=='List size out of allowed range'

def test_error_not_same_elem_size_on_get_cp() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size1=4
  item_size2=3
  min_list_size=1
  max_list_size=50
  schema1=((min_list_size, max_list_size), item_size1)
  schema2=((min_list_size, max_list_size), item_size2)
  li1=LI(input_len1, schema1)
  li2=LI(input_len2, schema2)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI.get_cp(li1, li2)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal gen size'

def test_error_not_same_min_list_on_get_cp() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size1=1
  min_list_size2=2
  max_list_size=50
  schema1=((min_list_size1, max_list_size), item_size)
  schema2=((min_list_size2, max_list_size), item_size)
  li1=LI(input_len1, schema1)
  li2=LI(input_len2, schema2)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI.get_cp(li1, li2)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal min list size'

def test_error_not_same_max_list_on_get_cp() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=1
  max_list_size1=50
  max_list_size2=5
  schema1=((min_list_size, max_list_size1), item_size)
  schema2=((min_list_size, max_list_size2), item_size)
  li1=LI(input_len1, schema1)
  li2=LI(input_len2, schema2)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI.get_cp(li1, li2)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal max list size'

def test_error_not_same_elem_size_on_crossover() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size1=4
  item_size2=3
  min_list_size=1
  max_list_size=50
  schema1=((min_list_size, max_list_size), item_size1)
  schema2=((min_list_size, max_list_size), item_size2)
  li1=LI(input_len1, schema1)
  li2=LI(input_len2, schema2)
  dummy_cp=(1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI.crossover(li1, li2, dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal gen size'

def test_error_not_same_min_list_on_crossover() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size1=1
  min_list_size2=2
  max_list_size=50
  schema1=((min_list_size1, max_list_size), item_size)
  schema2=((min_list_size2, max_list_size), item_size)
  li1=LI(input_len1, schema1)
  li2=LI(input_len2, schema2)
  dummy_cp=(1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI.crossover(li1, li2, dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal min list size'

def test_error_not_same_max_list_on_crossover() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=1
  max_list_size1=50
  max_list_size2=5
  schema1=((min_list_size, max_list_size1), item_size)
  schema2=((min_list_size, max_list_size2), item_size)
  li1=LI(input_len1, schema1)
  li2=LI(input_len2, schema2)
  dummy_cp=(1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI.crossover(li1, li2, dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal max list size'

def test_error_not_same_elem_size_on_create() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size1=4
  item_size2=3
  min_list_size=1
  max_list_size=50
  schema1=((min_list_size, max_list_size), item_size1)
  schema2=((min_list_size, max_list_size), item_size2)
  li1=LI(input_len1, schema1)
  li2=LI(input_len2, schema2)
  dummy_cp=(1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI(li1, li2, cross_point=dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal gen size'

def test_error_not_same_min_list_on_create() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size1=1
  min_list_size2=2
  max_list_size=50
  schema1=((min_list_size1, max_list_size), item_size)
  schema2=((min_list_size2, max_list_size), item_size)
  li1=LI(input_len1, schema1)
  li2=LI(input_len2, schema2)
  dummy_cp=(1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI(li1, li2, cross_point=dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal min list size'

def test_error_not_same_max_list_on_create() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=1
  max_list_size1=50
  max_list_size2=5
  schema1=((min_list_size, max_list_size1), item_size)
  schema2=((min_list_size, max_list_size2), item_size)
  li1=LI(input_len1, schema1)
  li2=LI(input_len2, schema2)
  dummy_cp=(1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI(li1, li2, cross_point=dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal max list size'


mark__test_error_too_short_on_create_from_two=pytest.mark.parametrize(
  'cp',
  [
    (0, 6),
    (1, 7),
    (2, 8),
    (3, 9),
    (0, 9),
  ],
)
@mark__test_error_too_short_on_create_from_two
def test_error_too_short_on_create_from_two(cp: tuple[int, int]) -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema) # xxx xxx
  li2=LI(input_len2, schema) # xxx xxx xxx

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI(li1, li2, cross_point=cp)

  # results
  assert str(excinfo.value)=='Solution too short'

mark__test_error_too_short_on_crossover=pytest.mark.parametrize(
  'cp',
  [
    # first too short
    (0, 6),
    (1, 7),
    (2, 8),
    (3, 9),
    (0, 9),
    # second too short
    (3, 0),
    (4, 1),
    (5, 2),
    (6, 3),
    (6, 0),
  ],
)
@mark__test_error_too_short_on_crossover
def test_error_too_short_on_crossover(cp: tuple[int, int]) -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema) # xxx xxx
  li2=LI(input_len2, schema) # xxx xxx xxx

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI.crossover(li1, li2, cp)

  # results
  assert str(excinfo.value)=='Solution too short'

mark__test_error_outside_range_on_create=pytest.mark.parametrize(
  'cp',
  [
    (-1, 2),
    (7, 1),
    (1, 10),
    (2, -1),
  ],
)
@mark__test_error_outside_range_on_create
def test_error_outside_range_on_create(cp: tuple[int, int]) -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema) # xxx xxx
  li2=LI(input_len2, schema) # xxx xxx xxx

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI(li1, li2, cross_point=cp)

  # results
  assert str(excinfo.value)=='Cross point is outside of solution'

mark__test_error_outside_range_on_crossover=pytest.mark.parametrize(
  'cp',
  [
    (-1, 2),
    (7, 1),
    (1, 10),
    (2, -1),
  ],
)
@mark__test_error_outside_range_on_crossover
def test_error_outside_range_on_crossover(cp: tuple[int, int]) -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema) # xxx xxx
  li2=LI(input_len2, schema) # xxx xxx xxx

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI.crossover(li1, li2, cp)

  # results
  assert str(excinfo.value)=='Cross point is outside of solution'

mark__test_error_offset_on_create=pytest.mark.parametrize(
  'cp',
  [
    (2, 6),
    (1, 6),
    (0, 8),
    (1, 8),
    (0, 7),
    (2, 7),
  ],
)
@mark__test_error_offset_on_create
def test_error_offset_on_create(cp: tuple[int, int]) -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema) # xxx xxx
  li2=LI(input_len2, schema) # xxx xxx xxx

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI(li1, li2, cross_point=cp)

  # results
  assert str(excinfo.value)=='Cross points\' offsets are not equal'

mark__test_error_offset_on_crossover=pytest.mark.parametrize(
  'cp',
  [
    (2, 6),
    (1, 6),
    (0, 8),
    (1, 8),
    (0, 7),
    (2, 7),
  ],
)
@mark__test_error_offset_on_crossover
def test_error_offset_on_crossover(cp: tuple[int, int]) -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema) # xxx xxx
  li2=LI(input_len2, schema) # xxx xxx xxx

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI.crossover(li1, li2, cp)

  # results
  assert str(excinfo.value)=='Cross points\' offsets are not equal'

mark__test_error_illegal_argument_on_create=pytest.mark.parametrize(
  'func_args_kwargs',
  [
    lambda li1, li2, input_len, schema: ((li1, schema), {'cross_point': (1, 1)}),
    lambda li1, li2, input_len, schema: ((input_len, li2), {'cross_point': (1, 1)}),
    lambda li1, li2, input_len, schema: ((li1, li2), {}),
    lambda li1, li2, input_len, schema: ((li1, li2), {'cross_point': None}),
    lambda li1, li2, input_len, schema: ((li1, schema), {}),
    lambda li1, li2, input_len, schema: ((li1, schema), {'cross_point': None}),
    lambda li1, li2, input_len, schema: ((input_len, li2), {}),
    lambda li1, li2, input_len, schema: ((input_len, li2), {'cross_point': None}),
  ],
)
@mark__test_error_illegal_argument_on_create
def test_error_illegal_argument_on_create(
  func_args_kwargs: t.Callable[
    [LI, LI, int, tuple[tuple[int, int], int]],
    tuple[tuple[t.Any, ...], dict[str, t.Any]]
  ],
) -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema)
  li2=LI(input_len2, schema)
  args, kwargs=func_args_kwargs(li1, li2, input_len1, schema)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=LI(*args, **kwargs)

  # results
  assert str(excinfo.value)=='Illegal argument options'
