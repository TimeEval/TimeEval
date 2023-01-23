import unittest

from timeeval.params import FullParameterGrid, IndependentParameterGrid


class TestParameterSearch(unittest.TestCase):

    def setUp(self) -> None:
        self.params = {"a": [1, 2], "b": [True, False], "c": ["auto", "tanh"]}

    def test_full_parameter_grid(self):
        grid = FullParameterGrid(self.params)
        self.assertListEqual(list(grid), [
            {"a": 1, "b": True, "c": "auto"},
            {"a": 1, "b": True, "c": "tanh"},
            {"a": 1, "b": False, "c": "auto"},
            {"a": 1, "b": False, "c": "tanh"},
            {"a": 2, "b": True, "c": "auto"},
            {"a": 2, "b": True, "c": "tanh"},
            {"a": 2, "b": False, "c": "auto"},
            {"a": 2, "b": False, "c": "tanh"}
        ])
        self.assertEqual(grid[1], {"a": 1, "b": True, "c": "tanh"})

    def test_full_parameter_grid_wrong_input(self):
        with self.assertRaises(TypeError) as ex:
            FullParameterGrid([{"a": [1, 2]}])
        self.assertRegex(str(ex.exception), "[P|p]lease use a.*IndependentParameterGrid.")

        with self.assertRaises(TypeError) as ex:
            FullParameterGrid("a")
        self.assertRegex(str(ex.exception), "[P|p]arameter grid is not a dict")

    def test_independent_parameter_grid(self):
        grid = IndependentParameterGrid(self.params)
        self.assertListEqual(list(grid), [
            {"a": 1},
            {"a": 2},
            {"b": True},
            {"b": False},
            {"c": "auto"},
            {"c": "tanh"}
        ])
        self.assertEqual(grid[2], {"b": True})

    def test_independent_parameter_grid_with_defaults(self):
        grid = IndependentParameterGrid(self.params, {"a": 0, "c": "auto"})
        self.assertListEqual(list(grid), [
            {"a": 1, "c": "auto"},
            {"a": 2, "c": "auto"},
            {"a": 0, "b": True, "c": "auto"},
            {"a": 0, "b": False, "c": "auto"},
            {"a": 0, "c": "auto"},
            {"a": 0, "c": "tanh"}
        ])

    def test_independent_parameter_grid_wrong_input(self):
        with self.assertRaises(TypeError) as ex1:
            IndependentParameterGrid([{"a": [1, 2]}])
        self.assertRegex(str(ex1.exception), "[P|p]arameter grid is not a dict")

        with self.assertRaises(TypeError) as ex2:
            IndependentParameterGrid({"a": [1, 2]}, default_params=[1])
        self.assertRegex(str(ex2.exception), "[D|d]efault parameters is not a dict")

    def test_independent_parameter_grid_non_value_list(self):
        grid = IndependentParameterGrid({"a": [1, 2], "b": True})
        self.assertListEqual(list(grid), [{"a": 1}, {"a": 2}, {"b": True}])

    def test_independent_parameter_grid_empty(self):
        grid = IndependentParameterGrid({}, {"a": 0, "c": "auto"})
        self.assertListEqual(list(grid), [{"a": 0, "c": "auto"}])

    def test_independent_parameter_grid_list_in_defaults(self):
        with self.assertRaises(TypeError) as ex1:
            IndependentParameterGrid(self.params, {"a": [0, 1], "c": "auto"})
        self.assertRegex(str(ex1.exception), "Only fixed values are allowed for defaults")
