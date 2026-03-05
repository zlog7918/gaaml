from gaaml.classes.Individual import Individual as I

class I_Test(I[str]):
  def __init__(self, gen: str) -> None:
    super().__init__(gen)
  def mutate(self) -> None: ...
def test_create() -> None:
  dummy_gen='test_text'
  i=I_Test(dummy_gen)
  assert isinstance(i.gen, str)
  assert i.gen==dummy_gen
