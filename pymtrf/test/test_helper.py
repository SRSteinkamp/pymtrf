#import context
import pymtrf
from ..helper import regularized_regression_fit
import numpy as np
import warnings
from .simulate_test_data import build_test_data

def test_lag_builder_positive_lags():
    # Test lag_builder for the creation of a positive lag vector
    lags = pymtrf.lag_builder(1, 4)
    assert np.all(lags == [1, 2, 3, 4])


def test_lag_builder_negative_lags():
    # Test lag_builder for the creation of a negative lag vector, starting with
    # a negative value
    lags = pymtrf.lag_builder(-2, 2)
    assert np.all(lags == [-2, -1, 0, 1, 2])


def test_lag_builder_negative_lags_reverse():
    # Test lag_builder for the creation of a negative lag vector, starting with
    # a positive value
    lags = pymtrf.lag_builder(2, -2)
    assert np.all(lags == [2, 1, 0, -1, -2])


def test_lag_builder_starting_from_zero():
    # Test lag_builder for the creation of a negative lag vector, starting with
    # a positive value
    lags = pymtrf.lag_builder(0, 3)
    assert np.all(lags == [0, 1, 2, 3])


def test_lag_builder_only_zero():
    # Test lag_builder for the creation of a negative lag vector, starting with
    # a positive value
    lags = pymtrf.lag_builder(0, 0)
    assert np.all(lags == [0])


def test_quadratic_regularization_3():
    m_mat = pymtrf.quadratic_regularization(3)
    test_mat = np.array([[1, -1, 0], [-1, 2, -1], [0, -1, 1]])
    assert np.all(m_mat == test_mat)


def test_quadratic_regularization_5():
    m_mat = pymtrf.quadratic_regularization(5)
    test_mat = np.array([[1, -1, 0, 0, 0], [-1, 2, -1, 0, 0],
                         [0, -1, 2, -1, 0], [0, 0, -1, 2, -1],
                         [0, 0, 0, -1, 1]])
    assert np.all(m_mat == test_mat)


def test_fit_regression_warning_x(recwarn):

    X = np.ones((10, 20))
    try:
        regularized_regression_fit(X, np.ones((10,1)), np.ones((20,20)))
    except:
        pass

    assert len(recwarn) == 1

    l = recwarn.pop(UserWarning)
    assert issubclass(l.category, UserWarning)
    assert str(l.message) == (f'X: more features {X.shape[1]}' +
                      f' than samples {X.shape[0]}, check input dimensions!')

def test_fit_regression_warning_y(recwarn):

    X = np.ones((5, 1))
    y = np.ones((5, 6))
    try:
        regularized_regression_fit(X, y, np.ones((5,5)))
    except:
        pass

    assert len(recwarn) == 1

    l = recwarn.pop(UserWarning)
    assert issubclass(l.category, UserWarning)
    assert str(l.message) == (f'y: more features {y.shape[1]}' +
                      f' than samples {y.shape[0]}, check input dimensions!')


def test_fit_regression_warning_Xintercept(recwarn):

    X = np.zeros((5, 2))
    y = np.ones((5, 2))
    try:
        regularized_regression_fit(X, y, np.ones((5,5)))
    except:
        pass

    assert len(recwarn) == 1

    l = recwarn.pop(UserWarning)
    assert issubclass(l.category, UserWarning)
    assert str(l.message) == ('Please check, whether an intercept term has been added!')



def test_create_test_data_x():
    x_shape = np.array([64 * 8, 5])
    y_shape = np.array([64 * 8, 6])
    model_shape = np.array([5, 9, 6])
    x, _, _ = build_test_data()
    assert np.all(x.shape == x_shape)


def test_create_test_data_model():
    x_shape = np.array([64 * 8, 5])
    y_shape = np.array([64 * 8, 6])
    model_shape = np.array([5, 9, 6])
    _, model, _ = build_test_data()

    assert np.all(model.shape == model_shape)


def test_create_test_data_y():
    x_shape = np.array([64 * 8, 5])
    y_shape = np.array([64 * 8, 6])
    model_shape = np.array([5, 9, 6])
    _, _, y =build_test_data()

    assert np.all(y.shape == y_shape)
