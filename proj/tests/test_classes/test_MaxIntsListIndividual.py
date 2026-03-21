import pytest
from gaaml.classes.ListIndividual import ListIndividual as _LI
from gaaml.classes.MaxIntsListIndividual import MaxIntsListIndividual as MILI

def test_create() -> None:
  # values
  input_len=5
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=1
  max_list_size=50
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )

  for _ in range(50): # random process
    # test
    mili=MILI(input_len, schema)

    # results
    assert isinstance(mili.gen, _LI)
    assert isinstance(mili.gen.gen, bytearray)
    assert len(mili.gen.gen)==input_len*item_size
    assert isinstance(mili.fenotype, list)
    assert len(mili.fenotype)==input_len

def test_mutate() -> None:
  # values
  input_len=5
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=1
  max_list_size=50
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili=MILI(input_len, schema)

  for _ in range(50): # random process
    oryg_gen=mili.gen.gen[:]

    # test
    mili.mutate()

    # results
    assert mili.gen.gen!=oryg_gen
    assert isinstance(mili.fenotype, list)
    assert len(mili.fenotype)==input_len

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
  min_elem_val=1
  max_elem_val=6
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len1, schema)
  mili2=MILI(input_len2, schema)

  for _ in range(50): # random process
    # test
    cp=MILI.get_cp(mili1, mili2)

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
  min_elem_val=1
  max_elem_val=6
  min_list_size=1
  max_list_size=50
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len1, schema)
  mili1.gen._gen=bytearray('0 1 1  1 1 1'.replace(' ', '').encode())
  mili2=MILI(input_len2, schema)
  mili2.gen._gen=bytearray('1 1 0  0 1 0  0 0 1'.replace(' ', '').encode())

  for cp, expected_str, expected_arr in (
    ((4, 7), '0 1 1  1 0 1', [4, 6]),
    ((4, 4), '0 1 1  1 1 0  0 0 1', [4, 5, 2]),
    ((4, 1), '0 1 1  1 1 0  0 1 0  0 0 1', [4, 5, 3, 2]),
    ((1, 7), '0 0 1', [2]),
    ((6, 0), '0 1 1  1 1 1  1 1 0  0 1 0  0 0 1', [4, 6, 5, 3, 2]),
    ((6, 3), '0 1 1  1 1 1  0 1 0  0 0 1', [4, 6, 3, 2]),
    ((6, 9), '0 1 1  1 1 1', [4, 6]),
  ):
    # test
    mili=MILI(mili1, mili2, cross_point=cp)

    # results
    assert mili.gen.gen==bytearray(expected_str.replace(' ', '').encode())
    assert mili.fenotype==expected_arr

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
  min_elem_val=1
  max_elem_val=6
  min_list_size=1
  max_list_size=3
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len1, schema)
  mili1.gen._gen=bytearray('0 1 1  1 1 1'.replace(' ', '').encode())
  mili2=MILI(input_len2, schema)
  mili2.gen._gen=bytearray('1 1 0  0 1 0  0 0 1'.replace(' ', '').encode())

  for cp, expected_str, expected_arr in (
    ((4, 1), '0 1 1  1 1 0  0 1 0', [4, 5, 3]), # '0 1 1  1 1 0  0 1 0  0 0 1'
    ((6, 0), '0 1 1  1 1 1  1 1 0', [4, 6, 5]), # '0 1 1  1 1 1  1 1 0  0 1 0  0 0 1'
    ((6, 3), '0 1 1  1 1 1  0 1 0', [4, 6, 3]), # '0 1 1  1 1 1  0 1 0  0 0 1'
  ):
    # test
    mili=MILI(mili1, mili2, cross_point=cp)

    # results
    assert mili.gen.gen==bytearray(expected_str.replace(' ', '').encode())
    assert mili.fenotype==expected_arr

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
  min_elem_val=1
  max_elem_val=6
  min_list_size=1
  max_list_size=50
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len1, schema)
  mili1.gen._gen=bytearray('0 1 1  1 1 1'.replace(' ', '').encode())
  mili2=MILI(input_len2, schema)
  mili2.gen._gen=bytearray('1 1 0  0 1 0  0 0 1'.replace(' ', '').encode())

  for cp, (expected_str1, expected_str2), (expected_arr1, expected_arr2) in (
    ((4, 7), ('0 1 1  1 0 1', '1 1 0  0 1 0  0 1 1'), ([4, 6], [5, 3, 4])),
    ((4, 4), ('0 1 1  1 1 0  0 0 1', '1 1 0  0 1 1'), ([4, 5, 2], [5, 4])),
    ((4, 1), ('0 1 1  1 1 0  0 1 0  0 0 1', '1 1 1'), ([4, 5, 3, 2], [6])),
    ((1, 7), ('0 0 1', '1 1 0  0 1 0  0 1 1  1 1 1'), ([2], [5, 3, 4, 6])),
    # ((6, 0), ('0 1 1  1 1 1  1 1 0  0 1 0  0 0 1', ''), ([4, 6, 5, 3, 2], [])), # throws error
    ((6, 3), ('0 1 1  1 1 1  0 1 0  0 0 1', '1 1 0'), ([4, 6, 3, 2], [5])),
    ((6, 9), ('0 1 1  1 1 1', '1 1 0  0 1 0  0 0 1'), ([4, 6], [5, 3, 2])),
  ):
    # test
    mili_s=MILI.crossover(mili1, mili2, cp)

    # results
    assert isinstance(mili_s, tuple)
    assert mili_s[0].gen.gen==bytearray(expected_str1.replace(' ', '').encode())
    assert mili_s[1].gen.gen==bytearray(expected_str2.replace(' ', '').encode())
    assert mili_s[0].fenotype==expected_arr1
    assert mili_s[1].fenotype==expected_arr2

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
  min_elem_val=1
  max_elem_val=6
  min_list_size=1
  max_list_size=3
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len1, schema)
  mili1.gen._gen=bytearray('0 1 1  1 1 1'.replace(' ', '').encode())
  mili2=MILI(input_len2, schema)
  mili2.gen._gen=bytearray('1 1 0  0 1 0  0 0 1'.replace(' ', '').encode())

  for cp, (expected_str1, expected_str2), (expected_arr1, expected_arr2) in (
    ((4, 1), ('0 1 1  1 1 0  0 1 0', '1 1 1'), ([4, 5, 3], [6])), # ('0 1 1  1 1 0  0 1 0  0 0 1', '1 1 1')
    ((1, 7), ('0 0 1', '1 1 0  0 1 0  0 1 1'), ([2], [5, 3, 4])), # ('0 0 1', '1 1 0  0 1 0  0 1 1  1 1 1')
    ((6, 3), ('0 1 1  1 1 1  0 1 0', '1 1 0'), ([4, 6, 3], [5])), # ('0 1 1  1 1 1  0 1 0  0 0 1', '1 1 0')
  ):
    # test
    mili_s=MILI.crossover(mili1, mili2, cp)

    # results
    assert isinstance(mili_s, tuple)
    assert mili_s[0].gen.gen==bytearray(expected_str1.replace(' ', '').encode())
    assert mili_s[1].gen.gen==bytearray(expected_str2.replace(' ', '').encode())
    assert mili_s[0].fenotype==expected_arr1
    assert mili_s[1].fenotype==expected_arr2

def test_get_fenotype() -> None:
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
  input_len=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=1
  max_list_size=3
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili=MILI(input_len, schema)

  for b_str, expected in (
    ('0 0 0  1 1 0  1 1 1', [1, 5, 6]),
    ('0 0 1  1 0 0  1 0 1', [2, 5, 6]),
    ('0 1 1  0 1 0  0 0 0', [4, 3, 1]),
  ):
    mili.gen._gen=bytearray(b_str.replace(' ', '').encode())

    # test
    fenotype=MILI.get_fenotype(mili.gen, *mili.schema)

    # results
    assert fenotype==expected

def test_error_list_size_too_small_on_create() -> None:
  # values
  input_len=2
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=3
  max_list_size=7
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MILI(input_len, schema)

  # results
  assert str(excinfo.value)=='List size out of allowed range'

def test_error_list_size_too_big_on_create() -> None:
  # values
  input_len=9
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=3
  max_list_size=7
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MILI(input_len, schema)

  # results
  assert str(excinfo.value)=='List size out of allowed range'

def test_error_not_same_conf_on_get_cp() -> None:
  # values
  input_len=5
  schema1=((3, 7), (3, 1, 6))
  mili1=MILI(input_len, schema1)
  for schema2 in (
    ((3, 7), (3, 0, 6)),
    ((3, 7), (3, 2, 6)),
    ((3, 7), (3, 1, 5)),
    ((3, 7), (3, 1, 7)),
  ):
    mili2=MILI(input_len, schema2)

    # test
    with pytest.raises(ValueError) as excinfo:
      _=MILI.get_cp(mili1, mili2)

    # results
    print(schema2)
    assert str(excinfo.value)=='First and second solution do not have equal configuration'

def test_error_not_same_elem_size_on_get_cp() -> None:
  # values
  input_len=3
  item_size1=4
  item_size2=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=3
  max_list_size=7
  schema1=(
    (min_list_size, max_list_size),
    (item_size1, min_elem_val, max_elem_val),
  )
  schema2=(
    (min_list_size, max_list_size),
    (item_size2, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len, schema1)
  mili2=MILI(input_len, schema2)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MILI.get_cp(mili1, mili2)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal gen size'

def test_error_not_same_min_list_on_get_cp() -> None:
  # values
  input_len=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size1=1
  min_list_size2=2
  max_list_size=7
  schema1=(
    (min_list_size1, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  schema2=(
    (min_list_size2, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len, schema1)
  mili2=MILI(input_len, schema2)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MILI.get_cp(mili1, mili2)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal min list size'

def test_error_not_same_max_list_on_get_cp() -> None:
  # values
  input_len=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=1
  max_list_size1=50
  max_list_size2=5
  schema1=(
    (min_list_size, max_list_size1),
    (item_size, min_elem_val, max_elem_val),
  )
  schema2=(
    (min_list_size, max_list_size2),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len, schema1)
  mili2=MILI(input_len, schema2)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MILI.get_cp(mili1, mili2)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal max list size'

def test_error_not_same_conf_on_crossover() -> None:
  # values
  input_len=5
  schema1=((3, 7), (3, 1, 6))
  mili1=MILI(input_len, schema1)
  dummy_cp=(1, 1)
  for schema2 in (
    ((3, 7), (3, 0, 6)),
    ((3, 7), (3, 2, 6)),
    ((3, 7), (3, 1, 5)),
    ((3, 7), (3, 1, 7)),
  ):
    mili2=MILI(input_len, schema2)

    # test
    with pytest.raises(ValueError) as excinfo:
      _=MILI.crossover(mili1, mili2, dummy_cp)

    # results
    assert str(excinfo.value)=='First and second solution do not have equal configuration'

def test_error_not_same_elem_size_on_crossover() -> None:
  # values
  input_len=3
  item_size1=4
  item_size2=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=3
  max_list_size=7
  schema1=(
    (min_list_size, max_list_size),
    (item_size1, min_elem_val, max_elem_val),
  )
  schema2=(
    (min_list_size, max_list_size),
    (item_size2, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len, schema1)
  mili2=MILI(input_len, schema2)
  dummy_cp=(1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MILI.crossover(mili1, mili2, dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal gen size'

def test_error_not_same_min_list_on_crossover() -> None:
  # values
  input_len=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size1=1
  min_list_size2=2
  max_list_size=7
  schema1=(
    (min_list_size1, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  schema2=(
    (min_list_size2, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len, schema1)
  mili2=MILI(input_len, schema2)
  dummy_cp=(1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MILI.crossover(mili1, mili2, dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal min list size'

def test_error_not_same_max_list_on_crossover() -> None:
  # values
  input_len=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=1
  max_list_size1=50
  max_list_size2=5
  schema1=(
    (min_list_size, max_list_size1),
    (item_size, min_elem_val, max_elem_val),
  )
  schema2=(
    (min_list_size, max_list_size2),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len, schema1)
  mili2=MILI(input_len, schema2)
  dummy_cp=(1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MILI.crossover(mili1, mili2, dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal max list size'

def test_error_not_same_conf_on_create() -> None:
  # values
  input_len=5
  schema1=((3, 7), (3, 1, 6))
  mili1=MILI(input_len, schema1)
  dummy_cp=(1, 1)
  for schema2 in (
    ((3, 7), (3, 0, 6)),
    ((3, 7), (3, 2, 6)),
    ((3, 7), (3, 1, 5)),
    ((3, 7), (3, 1, 7)),
  ):
    mili2=MILI(input_len, schema2)

    # test
    with pytest.raises(ValueError) as excinfo:
      _=MILI(mili1, mili2, cross_point=dummy_cp)

    # results
    assert str(excinfo.value)=='First and second solution do not have equal configuration'

def test_error_not_same_elem_size_on_create() -> None:
  # values
  input_len=3
  item_size1=4
  item_size2=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=3
  max_list_size=7
  schema1=(
    (min_list_size, max_list_size),
    (item_size1, min_elem_val, max_elem_val),
  )
  schema2=(
    (min_list_size, max_list_size),
    (item_size2, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len, schema1)
  mili2=MILI(input_len, schema2)
  dummy_cp=(1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MILI(mili1, mili2, cross_point=dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal gen size'

def test_error_not_same_min_list_on_create() -> None:
  # values
  input_len=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size1=1
  min_list_size2=2
  max_list_size=7
  schema1=(
    (min_list_size1, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  schema2=(
    (min_list_size2, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len, schema1)
  mili2=MILI(input_len, schema2)
  dummy_cp=(1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MILI(mili1, mili2, cross_point=dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal min list size'

def test_error_not_same_max_list_on_create() -> None:
  # values
  input_len=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=1
  max_list_size1=50
  max_list_size2=5
  schema1=(
    (min_list_size, max_list_size1),
    (item_size, min_elem_val, max_elem_val),
  )
  schema2=(
    (min_list_size, max_list_size2),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len, schema1)
  mili2=MILI(input_len, schema2)
  dummy_cp=(1, 1)

  # test
  with pytest.raises(ValueError) as excinfo:
    _=MILI(mili1, mili2, cross_point=dummy_cp)

  # results
  assert str(excinfo.value)=='First and second solution do not have equal max list size'

def test_error_too_short_on_create_from_two() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=2
  max_list_size=10
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len1, schema) # xxx xxx
  mili2=MILI(input_len2, schema) # xxx xxx xxx

  for cp in (
    (0, 6),
    (1, 7),
    (2, 8),
    (3, 9),
    (0, 9),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=MILI(mili1, mili2, cross_point=cp)

    # results
    assert str(excinfo.value)=='Solution too short'

def test_error_too_short_on_crossover() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=2
  max_list_size=10
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len1, schema) # xxx xxx
  mili2=MILI(input_len2, schema) # xxx xxx xxx

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
      _=MILI.crossover(mili1, mili2, cp)

    # results
    assert str(excinfo.value)=='Solution too short'

def test_error_outside_range_on_create() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=2
  max_list_size=10
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len1, schema) # xxx xxx
  mili2=MILI(input_len2, schema) # xxx xxx xxx

  for cp in (
    (-1, 2),
    (7, 1),
    (1, 10),
    (2, -1),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=MILI(mili1, mili2, cross_point=cp)

    # results
    assert str(excinfo.value)=='Cross point is outside of solution'

def test_error_outside_range_on_crossover() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=2
  max_list_size=10
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len1, schema) # xxx xxx
  mili2=MILI(input_len2, schema) # xxx xxx xxx

  for cp in (
    (-1, 2),
    (7, 1),
    (1, 10),
    (2, -1),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=MILI.crossover(mili1, mili2, cp)

    # results
    assert str(excinfo.value)=='Cross point is outside of solution'

def test_error_offset_on_create() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=2
  max_list_size=10
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len1, schema) # xxx xxx
  mili2=MILI(input_len2, schema) # xxx xxx xxx

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
      _=MILI(mili1, mili2, cross_point=cp)

    # results
    assert str(excinfo.value)=='Cross points\' offsets are not equal'

def test_error_offset_on_crossover() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=2
  max_list_size=10
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len1, schema) # xxx xxx
  mili2=MILI(input_len2, schema) # xxx xxx xxx

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
      _=MILI.crossover(mili1, mili2, cp)

    # results
    assert str(excinfo.value)=='Cross points\' offsets are not equal'

def test_error_illegal_argument_on_create() -> None:
  # values
  input_len1=2
  input_len2=3
  item_size=3
  min_elem_val=1
  max_elem_val=6
  min_list_size=2
  max_list_size=10
  schema=(
    (min_list_size, max_list_size),
    (item_size, min_elem_val, max_elem_val),
  )
  mili1=MILI(input_len1, schema)
  mili2=MILI(input_len2, schema)

  for args, kwargs in (
    ((mili1, schema), {'cross_point': (1, 1)}),
    ((input_len1, mili2), {'cross_point': (1, 1)}),
    ((mili1, mili2), {}),
    ((mili1, mili2), {'cross_point': None}),
    ((mili1, schema), {}),
    ((mili1, schema), {'cross_point': None}),
    ((input_len1, mili2), {}),
    ((input_len1, mili2), {'cross_point': None}),
  ):
    # test
    with pytest.raises(ValueError) as excinfo:
      _=MILI(*args, **kwargs) # type: ignore

    # results
    assert str(excinfo.value)=='Illegal argument options'
