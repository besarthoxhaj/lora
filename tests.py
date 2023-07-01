import unittest
from data import TrainData
from model import Model


class TestData(unittest.TestCase):
  def setUp(self):
    self.data = TrainData()

  def test_tokenize_prompt(self):
    res = self.data[0:2]
    assert len(res["input_ids"]) == 2
    assert len(res["attention_mask"]) == 2
    assert len(res["labels"]) == 2


class TestModel(unittest.TestCase):
  def setUp(self):
    self.model = Model()

  def test_model(self):
    pass

if __name__ == '__main__':
  unittest.main()