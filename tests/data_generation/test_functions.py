import random
import pytest
import numpy as np
from unittest.mock import patch
from typing import List

from d2c.data_generation.functions import *


def test_f_linear():
    """Test the linear function with two parents and two coefficients."""
    parents = ["x", "y"]
    test_inputs = {"x": 0.5, "y": -1.0}

    with patch("random.uniform") as mock_random_uniform:
        mock_random_uniform.side_effect = [1.2, 1.5]

        linear_func = f_linear(parents)
        result = linear_func(**test_inputs)

        expected = 1.2 * 0.5 + 1.5 * (-1.0)

        assert result == expected, f"Expected {expected}, but got {result}"


def test_f_polynomial():
    """Test the polynomial function with two parents and two degrees."""
    parents = ["x", "y"]
    degrees = [1, 3]
    test_inputs = {"x": 0.5, "y": -1.0}

    with patch("random.uniform") as mock_random_uniform:

        # coefficients are parent/degree combinations
        # so second coefficient is parent 1 degree 3
        mock_random_uniform.side_effect = [1.2, 1.5, 1.8, 2.1]
        poly_func = f_polynomial(parents, degrees)
        result = poly_func(**test_inputs)

        expected = (
            1.2 * (0.5**1) + 1.8 * (-(1.0**1)) + 1.5 * ((0.5**3)) + 2.1 * (-(1.0**3))
        )

    assert result == expected, f"Expected {expected}, but got {result}"


def test_f_sigmoid():
    """Test the sigmoid function with two parents."""
    parents = ["x", "y"]
    test_inputs = {"x": 0.5, "y": -1.0}

    with patch("random.uniform") as mock_random_uniform:
        mock_random_uniform.side_effect = [1.2, 1.5, 0.5]

        sigmoid_func = f_sigmoid(parents)
        result = sigmoid_func(**test_inputs)

        expected = 1 / (1 + np.exp(-(0.5 + 1.2 * 0.5 + 1.5 * (-1.0))))

    assert result == expected, f"Expected {expected}, but got {result}"


def test_f_nonlinear():
    """Test the nonlinear function with two parents and a custom nonlinearity."""
    parents = ["x", "y"]
    test_inputs = {"x": 0.5, "y": -1.0}

    with patch("random.uniform") as mock_random_uniform:
        mock_random_uniform.side_effect = [1.2, 1.5]

        sin_func = f_nonlinear(parents, np.sin)
        result = sin_func(**test_inputs)

        expected = np.sin(1.2 * 0.5 + 1.5 * (-1.0))

    assert result == expected, f"Expected {expected}, but got {result}"


def test_f_nonlinear_custom():
    """Test the nonlinear function with two parents and a custom nonlinearity."""
    parents = ["x", "y"]
    test_inputs = {"x": 0.5, "y": -1.0}

    def custom_nonlinearity(x):
        return x**2

    with patch("random.uniform") as mock_random_uniform:
        mock_random_uniform.side_effect = [1.2, 1.5]

        custom_func = f_nonlinear(parents, custom_nonlinearity)
        result = custom_func(**test_inputs)

        expected = custom_nonlinearity(1.2 * 0.5 + 1.5 * (-1.0))

    assert result == expected, f"Expected {expected}, but got {result}"


def test_f_interaction():
    """Test the interaction function with two parents."""
    parents = ["x", "y"]
    test_inputs = {"x": 0.5, "y": -1.2}

    with patch("random.uniform") as mock_random_uniform:
        mock_random_uniform.side_effect = [1.2, 1.5, 1.6]

        interaction_func = f_interaction(parents)
        result = interaction_func(**test_inputs)

        expected = 1.2 * 0.5 + 1.5 * (-1.2) + 1.6 * 0.5 * (-1.2)

    assert result == expected, f"Expected {expected}, but got {result}"
