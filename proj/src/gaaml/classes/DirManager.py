import shutil
import tempfile
import typing as t
from pathlib import Path

class DirManager:
  __dir: tempfile.TemporaryDirectory[str]|Path
  def __init__(self, directory: Path|str|None=None) -> None:
    if directory is None:
      self.__dir=tempfile.TemporaryDirectory()
      return
    if isinstance(directory, str):
      directory=Path(directory)
    self.__dir=self.__validate_dir(directory)

  @staticmethod
  def __validate_dir(path: Path) -> Path:
    if not path.exists():
      path.mkdir(parents=True)
    if not path.is_dir():
      raise ValueError(f'Given path does not point to directory: {path.as_posix()}')
    if next(path.iterdir(), None) is not None:
      raise ValueError(f'Given directory is not empty: {path.as_posix()}')
    return path.resolve()
  
  @staticmethod
  def __is_tmp(dir: tempfile.TemporaryDirectory[str]|Path) -> t.TypeIs[tempfile.TemporaryDirectory[str]]:
    return isinstance(dir, tempfile.TemporaryDirectory)
  
  @property
  def is_tmp(self) -> bool:
    return self.__is_tmp(self.__dir)

  @property
  def path(self) -> Path:
    return (
      Path(self.__dir.name).resolve()
        if self.__is_tmp(self.__dir) else
      self.__dir
    )

  @path.setter
  def path(self, path: Path|str) -> None:
    if isinstance(path, str):
      path=Path(path)
    path=self.__validate_dir(path)
    if path==self.path:
      return
    if self.path in path.parents:
      raise ValueError(f'Given path is inside the current directory: {path.as_posix()}')
    posix_path=path.as_posix()
    for el in self.path.iterdir():
      shutil.move(el.as_posix(), posix_path)
    if isinstance(self.__dir, tempfile.TemporaryDirectory):
      self.__dir.cleanup()
    self.__dir=path
