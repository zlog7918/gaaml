import atexit
import shutil
import tempfile
import typing as t
from pathlib import Path

class DirManager:
  __dir: tempfile.TemporaryDirectory[str]|Path
  @t.overload
  def __init__(self, directory: None=None) -> None: ...
  @t.overload
  def __init__(self, directory: Path|str, *, has_to_be_empty: bool=True) -> None: ...
  def __init__(self, directory: Path|str|None=None, *, has_to_be_empty: bool=True) -> None:
    if directory is None:
      self.__dir=tempfile.TemporaryDirectory()
      self.__cleanup_ref=self.__cleanup
      atexit.register(self.__cleanup_ref)
      return
    if isinstance(directory, str):
      directory=Path(directory)
    self.__dir=self.__validate_dir(directory, has_to_be_empty)

  @staticmethod
  def __validate_dir(path: Path, has_to_be_empty: bool=True) -> Path:
    if not path.exists():
      path.mkdir(parents=True)
    if not path.is_dir():
      raise ValueError(f'Given path does not point to directory: {path}')
    if has_to_be_empty and next(path.iterdir(), None) is not None:
      raise ValueError(f'Given directory is not empty: {path}')
    return path.resolve()

  @property
  def is_tmp(self) -> bool:
    return isinstance(self.__dir, tempfile.TemporaryDirectory)

  @property
  def path(self) -> Path:
    return (
      Path(self.__dir.name).resolve()
        if isinstance(self.__dir, tempfile.TemporaryDirectory) else
      self.__dir
    )

  def __cleanup(self) -> None:
    if isinstance(self.__dir, tempfile.TemporaryDirectory):
      self.__dir.cleanup()
      atexit.unregister(self.__cleanup_ref)
      del self.__cleanup_ref
      self.__dir=Path(self.__dir.name)

  @staticmethod
  def __move(src: Path, dst: Path) -> None:
    try:
      src.rename(dst)
    except OSError:
      shutil.move(src, dst)

  @path.setter
  def path(self, path: Path|str) -> None:
    if isinstance(path, str):
      path=Path(path)
    curr_path=self.path
    path=self.__validate_dir(path)
    if path==curr_path:
      return
    if path.is_relative_to(curr_path):
      raise ValueError(f'Given path is inside the current directory: {path}')

    items=list(curr_path.iterdir())
    targets=[path/el.name for el in items]
    moved: list[tuple[Path, Path]]=[]
    try:
      for el, target in zip(items, targets):
        self.__move(el, target)
        moved.append((target, el))
    except Exception:
      for src, dst in reversed(moved):
        if src.exists():
          self.__move(src, dst)
      raise
    self.__cleanup()
    self.__dir=path
