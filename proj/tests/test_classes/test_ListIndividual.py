import pytest
from gaaml.classes.ListIndividual import ListIndividual as LI

def test_create() -> None:
  # values
  input_len=5
  item_size=3
  min_list_size=1
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)

  # test
  li=LI(input_len, schema)

  # results
  assert isinstance(li.gen, bytearray)
  assert len(li.gen)==input_len*item_size

def test_mutate() -> None:
  # values
  input_len=5
  item_size=3
  min_list_size=1
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
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

  for i in range(1, 50): # random process
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

def test_create_from_two() -> None:
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
  li1.gen=bytearray('0 1 1  1 1 1'.replace(' ', '').encode())
  li2=LI(input_len2, schema)
  li2.gen=bytearray('1 1 0  0 1 0  0 0 1'.replace(' ', '').encode())

  for cp, expected_str in (
    ((4, 7), '0 1 1  1 0 1'),
    ((4, 4), '0 1 1  1 1 0  0 0 1'),
    ((4, 1), '0 1 1  1 1 0  0 1 0  0 0 1'),
    ((1, 7), '0 0 1'),
    ((6, 0), '0 1 1  1 1 1  1 1 0  0 1 0  0 0 1'),
    ((6, 3), '0 1 1  1 1 1  0 1 0  0 0 1'),
    ((6, 9), '0 1 1  1 1 1'),
  ):
    # test
    li=LI(li1, li2, cross_point=cp)

    # results
    assert isinstance(li, LI)
    assert li.gen==bytearray(expected_str.replace(' ', '').encode())

def test_create_from_two_too_long() -> None:
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
  li1.gen=bytearray('0 1 1  1 1 1'.replace(' ', '').encode())
  li2=LI(input_len2, schema)
  li2.gen=bytearray('1 1 0  0 1 0  0 0 1'.replace(' ', '').encode())

  for cp, expected_str in (
    ((4, 1), '0 1 1  1 1 0  0 1 0'), # '0 1 1  1 1 0  0 1 0  0 0 1'
    ((6, 0), '0 1 1  1 1 1  1 1 0'), # '0 1 1  1 1 1  1 1 0  0 1 0  0 0 1'
    ((6, 3), '0 1 1  1 1 1  0 1 0'), # '0 1 1  1 1 1  0 1 0  0 0 1'
  ):
    # test
    li=LI(li1, li2, cross_point=cp)

    # results
    assert isinstance(li, LI)
    assert li.gen==bytearray(expected_str.replace(' ', '').encode())

def test_crossover() -> None:
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
  li1.gen=bytearray('0 1 1  1 1 1'.replace(' ', '').encode())
  li2=LI(input_len2, schema)
  li2.gen=bytearray('1 1 0  0 1 0  0 0 1'.replace(' ', '').encode())

  for cp, (expected_str1, expected_str2) in (
    ((4, 7), ('0 1 1  1 0 1', '1 1 0  0 1 0  0 1 1')),
    ((4, 4), ('0 1 1  1 1 0  0 0 1', '1 1 0  0 1 1')),
    ((4, 1), ('0 1 1  1 1 0  0 1 0  0 0 1', '1 1 1')),
    ((1, 7), ('0 0 1', '1 1 0  0 1 0  0 1 1  1 1 1')),
    # ((6, 0), ('0 1 1  1 1 1  1 1 0  0 1 0  0 0 1', '')), # throws error
    ((6, 3), ('0 1 1  1 1 1  0 1 0  0 0 1', '1 1 0')),
    ((6, 9), ('0 1 1  1 1 1', '1 1 0  0 1 0  0 0 1')),
  ):
    # test
    li_s=LI.crossover(li1, li2, cp)

    # results
    assert isinstance(li_s, tuple)
    assert li_s[0].gen==bytearray(expected_str1.replace(' ', '').encode())
    assert li_s[1].gen==bytearray(expected_str2.replace(' ', '').encode())

def test_crossover_too_long() -> None:
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
  li1.gen=bytearray('0 1 1  1 1 1'.replace(' ', '').encode())
  li2=LI(input_len2, schema)
  li2.gen=bytearray('1 1 0  0 1 0  0 0 1'.replace(' ', '').encode())

  for cp, (expected_str1, expected_str2) in (
    ((4, 1), ('0 1 1  1 1 0  0 1 0', '1 1 1')), # ('0 1 1  1 1 0  0 1 0  0 0 1', '1 1 1')
    ((1, 7), ('0 0 1', '1 1 0  0 1 0  0 1 1')), # ('0 0 1', '1 1 0  0 1 0  0 1 1  1 1 1')
    ((6, 3), ('0 1 1  1 1 1  0 1 0', '1 1 0')), # ('0 1 1  1 1 1  0 1 0  0 0 1', '1 1 0')
  ):
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

def test_error_too_short_on_create() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema) # xxx xxx
  li2=LI(input_len2, schema) # xxx xxx xxx

  for cp in (
    (0, 6),
    (1, 7),
    (2, 8),
    (3, 9),
    (0, 9),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=LI(li1, li2, cross_point=cp)

    # results
    assert str(excinfo.value)=='Solution too short'

def test_error_too_short_on_crossover() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema) # xxx xxx
  li2=LI(input_len2, schema) # xxx xxx xxx

  for cp in (
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
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=LI.crossover(li1, li2, cp)

    # results
    assert str(excinfo.value)=='Solution too short'

def test_error_outside_range_on_create() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema) # xxx xxx
  li2=LI(input_len2, schema) # xxx xxx xxx

  for cp in (
    (-1, 2),
    (7, 1),
    (1, 10),
    (2, -1),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=LI(li1, li2, cross_point=cp)

    # results
    assert str(excinfo.value)=='Cross point is outside of solution'

def test_error_outside_range_on_crossover() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema) # xxx xxx
  li2=LI(input_len2, schema) # xxx xxx xxx

  for cp in (
    (-1, 2),
    (7, 1),
    (1, 10),
    (2, -1),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=LI.crossover(li1, li2, cp)

    # results
    assert str(excinfo.value)=='Cross point is outside of solution'

def test_error_offset_on_create() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema) # xxx xxx
  li2=LI(input_len2, schema) # xxx xxx xxx

  for cp in (
    (2, 6),
    (1, 6),
    (0, 8),
    (1, 8),
    (0, 7),
    (2, 7),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=LI(li1, li2, cross_point=cp)

    # results
    assert str(excinfo.value)=='Cross points\' offsets are not equal'

def test_error_offset_on_crossover() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema) # xxx xxx
  li2=LI(input_len2, schema) # xxx xxx xxx

  for cp in (
    (2, 6),
    (1, 6),
    (0, 8),
    (1, 8),
    (0, 7),
    (2, 7),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=LI.crossover(li1, li2, cp)

    # results
    assert str(excinfo.value)=='Cross points\' offsets are not equal'

def test_error_illegal_argument_on_create() -> None:
  input_len1=2
  input_len2=3
  item_size=3
  min_list_size=2
  max_list_size=50
  schema=((min_list_size, max_list_size), item_size)
  li1=LI(input_len1, schema)
  li2=LI(input_len2, schema)
  for args, kwargs in (
    ((li1, schema), {'cross_point': (1, 1)}),
    ((input_len1, li2), {'cross_point': (1, 1)}),
    ((li1, li2), {}),
    ((li1, li2), {'cross_point': None}),
    ((li1, schema), {}),
    ((li1, schema), {'cross_point': None}),
    ((input_len1, li2), {}),
    ((input_len1, li2), {'cross_point': None}),
  ):
    with pytest.raises(ValueError) as excinfo:
      _=LI(*args, **kwargs) # type: ignore
    assert str(excinfo.value)=='Illegal argument options'
