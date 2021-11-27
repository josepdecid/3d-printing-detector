import os
import unittest

from transforms.random_background import RandomBackground


class TestRandomBackground(unittest.TestCase):
    def setUp(self) -> None:
        self.prep = RandomBackground(os.path.join('test', 'resources'))

    def test_tests_work_properly(self):
        self.assertTrue(True)