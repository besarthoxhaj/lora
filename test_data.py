import unittest
from data import TrainData


class TesData(unittest.TestCase):

  def setUp(self):
    print("asdfasdsafa", flush=True)
    self.data = TrainData()

  def test_tokenize_prompt(self):
    self.data[0:2]
    pass

if __name__ == '__main__':
  unittest.main()