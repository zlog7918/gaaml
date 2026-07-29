import pytest
import tempfile
import typing as t
from pathlib import Path
from gaaml.classes.DirManager import DirManager as DM

mark__test_create=pytest.mark.parametrize(
  ('path_to_args_kwargs', 'exp_flag'),
  [
    (fp_2akw, flag) for fp_2akw, flag in (
      (lambda p: ((), {}), True),
      (lambda p: ((None,), {}), True),
      *(f for kwargs in (
        {},
        {'has_to_be_empty': True},
        {'has_to_be_empty': False},
      ) for f in (
        (lambda p: ((p,), kwargs), False),
        (lambda p: ((str(p),), kwargs), False),
      ))
    )
  ]
)
@mark__test_create
def test_create(
  tmp_path: Path,
  path_to_args_kwargs: t.Callable[[Path], tuple[tuple[t.Any, ...], dict[str, bool]]],
  exp_flag: bool,
) -> None:
  # values
  args, kwargs=path_to_args_kwargs(tmp_path)

  # test
  dm=DM(*args, **kwargs)

  # results
  # private access: dm.__dir
  assert dm.is_tmp==exp_flag
  _dir=dm._DirManager__dir # type: ignore
  assert isinstance(_dir, tempfile.TemporaryDirectory)==exp_flag
  assert isinstance(dm.path, Path)
  assert dm.path.exists()
  assert dm.path.is_dir()
  assert dm.path.is_absolute()
  assert (dm.path==tmp_path.resolve()) ^ exp_flag

mark__test_create_create_dir=pytest.mark.parametrize(
  ('path_to_sub_path', 'kwargs'),
  [
    (fp_2sp, kwargs) for kwargs in (
      {},
      {'has_to_be_empty': True},
      {'has_to_be_empty': False},
    ) for fp_2sp in (
      lambda p: p/'dir',
      lambda p: str(p/'dir'),
      lambda p: p/'dir'/'sub',
      lambda p: str(p/'dir'/'sub'),
    )
  ]
)
@mark__test_create_create_dir
def test_create_create_dir(
  tmp_path: Path,
  path_to_sub_path: t.Callable[[Path], Path|str],
  kwargs: dict[str, bool],
) -> None:
  # values
  path=path_to_sub_path(tmp_path)
  path_P=Path(path)
  path_exists=path_P.exists()

  # test
  dm=DM(path, **kwargs)

  # results
  # private access: dm.__dir
  assert not path_exists
  assert path_P.exists()
  assert path_P.is_dir()
  _dir=dm._DirManager__dir # type: ignore
  assert isinstance(_dir, Path)
  ret_path=dm.path
  assert isinstance(ret_path, Path)
  assert ret_path==path_P.resolve()
  assert ret_path.exists()
  assert ret_path.is_dir()
  assert ret_path.is_absolute()

mark__test_create_resume_non_empty_dir=pytest.mark.parametrize(
  'path_to_dir_path',
  [
    lambda p: str(p),
    lambda p: p,
  ]
)
@mark__test_create_resume_non_empty_dir
def test_create_resume_non_empty_dir(
  tmp_path: Path,
  path_to_dir_path: t.Callable[[Path], Path|str],
) -> None:
  # values
  dir_path=path_to_dir_path(tmp_path)
  dir_Path=Path(dir_path)
  f=(dir_Path/'t.txt')
  f.write_text('data')

  # test
  dm=DM(dir_path, has_to_be_empty=False)

  # results
  assert not dm.is_tmp
  # private access: dm.__dir
  _dir=dm._DirManager__dir # type: ignore
  assert isinstance(_dir, Path)
  assert dm.path==dir_Path.resolve()
  assert f.exists()
  assert f.resolve().is_relative_to(dm.path)

mark__test_is_tmp=pytest.mark.parametrize(
  ('path_to_args', 'exp_flag'),
  [
    ((lambda p: ()), True),
    ((lambda p: (None,)), True),
    ((lambda p: (str(p),)), False),
    ((lambda p: (p,)), False),
  ]
)
@mark__test_is_tmp
def test_is_tmp(
  tmp_path: Path,
  path_to_args: t.Callable[[Path], tuple[t.Any, ...]],
  exp_flag: bool
) -> None:
  # values
  args=path_to_args(tmp_path)
  dm=DM(*args)

  # test
  is_tmp=dm.is_tmp

  # results
  # private access: dm.__dir
  _dir=dm._DirManager__dir # type: ignore
  assert is_tmp==exp_flag
  assert isinstance(_dir, tempfile.TemporaryDirectory)==exp_flag

mark__test_path_setter=pytest.mark.parametrize(
  ('path_to_args', 'path_to_path_to_set'),
  [
    (fp_2a, fp2p2s) for fp_2a in (
      lambda p: (),
      lambda p: (None,),
      lambda p: (p/'dir1',),
      lambda p: (str(p/'dir1'),),
    ) for fp2p2s in (
      lambda p: p/'dir2',
      lambda p: str(p/'dir2'),
    )
  ]
)
@mark__test_path_setter
def test_path_setter(
  tmp_path: Path,
  path_to_args: t.Callable[[Path], tuple[t.Any, ...]],
  path_to_path_to_set: t.Callable[[Path], Path|str],
) -> None:
  # values
  args=path_to_args(tmp_path)
  path=path_to_path_to_set(tmp_path)
  path_P=Path(path).resolve()
  dm=DM(*args)

  # test
  dm.path=path

  # results
  # private access: dm.__dir
  _dir=dm._DirManager__dir # type: ignore
  assert not dm.is_tmp
  assert isinstance(_dir, Path)
  assert isinstance(dm.path, Path)
  assert dm.path==path_P
  assert _dir==path_P

mark__test_temp_dir_is_deleted_on_reassignment=pytest.mark.parametrize(
  ('args', 'path_to_path_to_set'),
  [
    (args, fp2p2s) for args in (
      (),
      (None,),
    ) for _fp2p2s in (
      lambda p: p,
      lambda p: p/'new_dir',
      lambda p: p/'new_dir'/'new_sub',
    ) for fp2p2s in (
      lambda p: _fp2p2s(p),
      lambda p: str(_fp2p2s(p)),
    )
  ]
)
@mark__test_temp_dir_is_deleted_on_reassignment
def test_temp_dir_is_deleted_on_reassignment(
  tmp_path: Path,
  args: tuple[()]|tuple[None],
  path_to_path_to_set: t.Callable[[Path], Path|str],
):
  # values
  new_dir=path_to_path_to_set(tmp_path)
  new_dir_Path=Path(new_dir).resolve()
  dm=DM(*args)
  old_path=dm.path
  prev_is_tmp=dm.is_tmp

  # test
  dm.path=new_dir

  # results
  assert prev_is_tmp
  assert not dm.is_tmp
  assert dm.path==new_dir_Path
  assert new_dir_Path.exists()
  assert not old_path.exists() # old temp dir should be gone

mark__test_not_tmp_dir_is_cleaned_on_reassignment=pytest.mark.parametrize(
  ('path_to_args', 'path_to_path_to_set'),
  [
    (fp_2a, fp2p2s) for fp_2a in (
      lambda p: (p/'dir1',),
      lambda p: (str(p/'dir1'),),
    ) for _fp2p2s in (
      lambda p: p/'dir2',
      lambda p: p/'dir2'/'sub',
    ) for fp2p2s in (
      lambda p: _fp2p2s(p),
      lambda p: str(_fp2p2s(p)),
    )
  ]
)
@mark__test_not_tmp_dir_is_cleaned_on_reassignment
def test_not_tmp_dir_is_cleaned_on_reassignment(
  tmp_path: Path,
  path_to_args: t.Callable[[Path], tuple[Path|str]],
  path_to_path_to_set: t.Callable[[Path], Path|str],
):
  # values
  args=path_to_args(tmp_path)
  new_dir=path_to_path_to_set(tmp_path)
  path_P=Path(new_dir).resolve()
  dm=DM(*args)
  old_path=dm.path
  prev_is_tmp=dm.is_tmp

  # test
  dm.path=new_dir

  # results
  assert not prev_is_tmp
  assert not dm.is_tmp
  assert dm.path==path_P
  assert path_P.exists()
  assert old_path.exists() # old dir should not be gone
  assert next(old_path.iterdir(), None) is None # old dir should be empty

mark__test_dir_is_created_on_assignment=pytest.mark.parametrize(
  ('path_to_args', 'exp_is_tmp', 'path_to_path_to_set'),
  [
    (fp_2akw, flag, fp2p2s) for fp_2akw, flag in (
      (lambda p: (), True),
      (lambda p: (None,), True),
      (lambda p: (p/'dir',), False),
      (lambda p: (str(p/'dir'),), False),
    ) for fp2p2s in (
      lambda p: p/'new_dir',
      lambda p: str(p/'new_dir'),
    )
  ]
)
@mark__test_dir_is_created_on_assignment
def test_dir_is_created_on_assignment(
  tmp_path: Path,
  path_to_args: t.Callable[[Path], tuple[t.Any, ...]],
  exp_is_tmp: bool,
  path_to_path_to_set: t.Callable[[Path], Path|str],
):
  # values
  args=path_to_args(tmp_path)
  path=path_to_path_to_set(tmp_path)
  path_P=Path(path)
  path_exists=path_P.exists()
  dm=DM(*args)
  old_path=dm.path
  prev_is_tmp=dm.is_tmp

  # test
  dm.path=path

  # results
  assert not path_exists
  assert not dm.is_tmp
  assert dm.path==path_P
  assert path_P.exists()

  assert prev_is_tmp==exp_is_tmp
  assert old_path.exists()!=exp_is_tmp

mark__test_path_setter_moves_contents=pytest.mark.parametrize(
  ('path_to_src', 'path_to_dst'),
  [
    (lambda p: str(p/'src'), lambda p: str(p/'dst')),
    (lambda p: str(p/'src'), lambda p: p/'dst'),
    (lambda p: p/'src', lambda p: str(p/'dst')),
    (lambda p: p/'src', lambda p: p/'dst'),
  ]
)
@mark__test_path_setter_moves_contents
def test_path_setter_moves_contents(
  tmp_path: Path,
  path_to_src: t.Callable[[Path], Path|str],
  path_to_dst: t.Callable[[Path], Path|str],
):
  # values
  src=path_to_src(tmp_path)
  dst=path_to_dst(tmp_path)

  dm=DM(src)
  src_file=Path(src)/'t.txt'
  dst_file=Path(dst)/'t.txt'
  src_file.write_text('hello')
  dst_file_exists=dst_file.exists()

  # test
  dm.path=dst

  # results
  assert dm.path==Path(dst).resolve()
  assert not dst_file_exists
  assert dst_file.exists()
  assert dst_file.read_text()=='hello'
  assert not src_file.exists()

def test_move_fallback_used(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  # values
  src=tmp_path/'src'
  dst=tmp_path/'dst'
  dm=DM(src)
  f=(src/'file.txt')
  f.write_text('data')

  def fake_rename(self, target):
    raise OSError()
  monkeypatch.setattr(Path, "rename", fake_rename)

  # test
  dm.path=dst

  # results
  assert (dst/'file.txt').exists()
  assert not f.exists()

mark__test_path_setter_same_path=pytest.mark.parametrize(
  'path_to_dir_path',
  [
    lambda p: str(p/'dir'),
    lambda p: p/'dir',
  ]
)
@mark__test_path_setter_same_path
def test_path_setter_same_path(
  tmp_path: Path,
  path_to_dir_path: t.Callable[[Path], Path|str],
):
  # values
  dir_path=path_to_dir_path(tmp_path)
  dm=DM(dir_path)

  # test
  before=dm.path
  dm.path=dir_path

  # results
  assert dm.path is before

def test_partial_move_rollback(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
):
  # values
  src=tmp_path/'src'
  dst=tmp_path/'dst'
  dm=DM(src)

  f1=src/'a.txt'
  f2=src/'b.txt'
  f1.write_text('a')
  f2.write_text('b')

  # Private access: DM.__move
  original_move: t.Callable[[Path, Path], None]
  original_move=DM._DirManager__move # type: ignore
  counter={'i': 0}

  @staticmethod
  def flaky_move(s: Path, d: Path) -> None:
    counter['i']+=1
    if counter['i']==2:
      raise RuntimeError("fail mid-move")
    return original_move(s, d)

  monkeypatch.setattr(DM, "_DirManager__move", flaky_move)

  # test
  with pytest.raises(RuntimeError):
    dm.path=dst

  # results
  assert (src/'a.txt').exists()
  assert (src/'b.txt').exists()
  assert not dst.exists() or next(dst.iterdir(), None) is None

mark__test_error_create_path_is_file=pytest.mark.parametrize(
  'path_to_file_path',
  [
    lambda p: str(p/'t.txt'),
    lambda p: p/'t.txt',
  ]
)
@mark__test_error_create_path_is_file
def test_error_create_path_is_file(
  tmp_path: Path,
  path_to_file_path: t.Callable[[Path], Path|str]
) -> None:
  # values
  file_path=path_to_file_path(tmp_path)
  file_Path=Path(file_path)
  file_Path.write_text('data')

  # test
  with pytest.raises(ValueError) as excinfo:
    _=DM(file_path)

  # results
  assert str(excinfo.value)==f'Given path does not point to directory: {file_Path.resolve()}'

mark__test_error_create_resume_path_is_file=pytest.mark.parametrize(
  'path_to_file_path',
  [
    lambda p: str(p/'t.txt'),
    lambda p: p/'t.txt',
  ]
)
@mark__test_error_create_resume_path_is_file
def test_error_create_resume_path_is_file(
  tmp_path: Path,
  path_to_file_path: t.Callable[[Path], Path|str],
) -> None:
  # values
  file_path=path_to_file_path(tmp_path)
  file_Path=Path(file_path)
  file_Path.write_text('data')

  # test
  with pytest.raises(ValueError) as excinfo:
    _=DM(file_path, has_to_be_empty=True)

  # results
  assert str(excinfo.value)==f'Given path does not point to directory: {file_Path.resolve()}'

mark__test_error_create_if_directory_not_empty=pytest.mark.parametrize(
  ('path_to_dir_path', 'kwargs'),
  [
    (fp_2dp, kwargs) for kwargs in (
      {},
      {'has_to_be_empty': True},
    ) for fp_2dp in (
      lambda p: p,
      lambda p: str(p),
    )
  ]
)
@mark__test_error_create_if_directory_not_empty
def test_error_create_if_directory_not_empty(
  tmp_path: Path,
  path_to_dir_path: t.Callable[[Path], Path|str],
  kwargs: dict[str, bool],
):
  # values
  dir_path=path_to_dir_path(tmp_path)
  dir_Path=Path(dir_path)
  (dir_Path/'t.txt').write_text('data')

  # test
  with pytest.raises(ValueError) as excinfo:
    _=DM(dir_path, **kwargs)

  # results
  assert str(excinfo.value)==f'Given directory is not empty: {dir_Path.resolve()}'

mark__test_error_path_setter_path_is_file=pytest.mark.parametrize(
  ('path_to_args', 'path_to_file_path'),
  [
    (fp_2a, fp2p) for fp_2a in (
      lambda p: (),
      lambda p: (None,),
      lambda p: (p/'dir',),
      lambda p: (str(p/'dir'),),
    ) for fp2p in (
      lambda p: p/'t.txt',
      lambda p: str(p/'t.txt'),
    )
  ]
)
@mark__test_error_path_setter_path_is_file
def test_error_path_setter_path_is_file(
  tmp_path: Path,
  path_to_args: t.Callable[[Path], tuple[t.Any, ...]],
  path_to_file_path: t.Callable[[Path], Path|str]
) -> None:
  # values
  args=path_to_args(tmp_path)
  file_path=path_to_file_path(tmp_path)
  file_Path=Path(file_path)
  file_Path.write_text('data')
  dm=DM(*args)

  # test
  with pytest.raises(ValueError) as excinfo:
    dm.path=file_path

  # results
  assert str(excinfo.value)==f'Given path does not point to directory: {file_Path.resolve()}'

mark__test_error_path_setter_if_directory_not_empty=pytest.mark.parametrize(
  ('path_to_args', 'path_to_dir_path'),
  [
    (fp_2a, fp2p2dp) for fp_2a in (
      lambda p: (),
      lambda p: (None,),
      lambda p: (p/'dir',),
      lambda p: (str(p/'dir'),),
    ) for fp2p2dp in (
      lambda p: p/'dir2',
      lambda p: str(p/'dir2'),
    )
  ]
)
@mark__test_error_path_setter_if_directory_not_empty
def test_error_path_setter_if_directory_not_empty(
  tmp_path: Path,
  path_to_args: t.Callable[[Path], tuple[t.Any, ...]],
  path_to_dir_path: t.Callable[[Path], Path|str],
):
  # values
  args=path_to_args(tmp_path)
  dir_path=path_to_dir_path(tmp_path)
  dir_Path=Path(dir_path)
  dir_Path.mkdir()
  (dir_Path/'t.txt').write_text('data')
  dm=DM(*args)

  # test
  with pytest.raises(ValueError) as excinfo:
    dm.path=dir_path

  # results
  assert str(excinfo.value)==f'Given directory is not empty: {dir_Path.resolve()}'

mark__test_error_path_setter_directory_is_inside_current_dir=pytest.mark.parametrize(
  ('path_to_path', 'path_to_subpath'),
  [
    (lambda p: str(p/'dir'), lambda p: str(p/'dir'/'sub')),
    (lambda p: str(p/'dir'), lambda p: p/'dir'/'sub'),
    (lambda p: p/'dir', lambda p: str(p/'dir'/'sub')),
    (lambda p: p/'dir', lambda p: p/'dir'/'sub'),
  ]
)
@mark__test_error_path_setter_directory_is_inside_current_dir
def test_error_path_setter_directory_is_inside_current_dir(
  tmp_path: Path,
  path_to_path: t.Callable[[Path], Path|str],
  path_to_subpath: t.Callable[[Path], Path|str],
):
  # values
  dir_path=path_to_path(tmp_path)
  subdir_path=path_to_subpath(tmp_path)
  subdir_Path=Path(subdir_path)
  dm=DM(dir_path)

  # test
  with pytest.raises(ValueError) as excinfo:
    dm.path=subdir_path

  # results
  assert str(excinfo.value)==f'Given path is inside the current directory: {subdir_Path.resolve()}'

mark__test_error_change_is_tmp=pytest.mark.parametrize(
  'path_to_args',
  [
    lambda p: (),
    lambda p: (None,),
    lambda p: (str(p),),
    lambda p: (p,),
  ]
)
@mark__test_error_change_is_tmp
def test_error_change_is_tmp(
  tmp_path: Path,
  path_to_args: t.Callable[[Path], tuple[t.Any, ...]],
) -> None:
  # values
  args=path_to_args(tmp_path)
  dm=DM(*args)
  exp_err={
    'can\'t set attribute \'is_tmp\'',
    'property \'is_tmp\' of \'DirManager\' object has no setter'
  }

  # test
  with pytest.raises(AttributeError) as excinfo1:
    dm.is_tmp=True # type: ignore
  with pytest.raises(AttributeError) as excinfo2:
    dm.is_tmp=False # type: ignore

  # results
  assert str(excinfo1.value) in exp_err
  assert str(excinfo2.value) in exp_err
