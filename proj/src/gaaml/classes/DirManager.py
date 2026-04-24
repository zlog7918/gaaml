import atexit
import shutil
import tempfile
import typing as t
from pathlib import Path

class DirManager:
  __dir: tempfile.TemporaryDirectory[str]|Path
  def __init__(self, directory: Path|str|None=None) -> None:
    if directory is None:
      self.__dir=tempfile.TemporaryDirectory()
      atexit.register(self.__cleanup)
      return
    if isinstance(directory, str):
      directory=Path(directory)
    self.__dir=self.__validate_dir(directory)

  @staticmethod
  def __validate_dir(path: Path) -> Path:
    if not path.exists():
      path.mkdir(parents=True)
    if not path.is_dir():
      raise ValueError(f'Given path does not point to directory: {path}')
    if next(path.iterdir(), None) is not None:
      raise ValueError(f'Given directory is not empty: {path}')
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

  def __cleanup(self) -> None:
    if self.__is_tmp(self.__dir):
      self.__dir.cleanup()
      atexit.unregister(self.__cleanup)
      self.__dir=Path(self.__dir.name)

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
    for el in curr_path.iterdir():
      shutil.move(el, path)
    self.__cleanup()
    self.__dir=path
