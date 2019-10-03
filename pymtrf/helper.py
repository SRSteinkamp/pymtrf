import numpy as np
from scipy import linalg
import warnings


def lag_builder(time_min, time_max):
    """Build the lags for the lag_generator function. Basically the indices of
    the time lags (including the starting and stopping points) of the data
    matrix.

    Parameters
    ----------
    time_min : np.int
        The starting index of the matrix as integer.
    time_max : np.int
        The stopping index of the matrix as integer.

    Returns
    -------
    lag_vector : numpy.ndarray, shape (np.abs(time_max) + np.abs(time_min) + 1,)
        A numpy array including all the lags.
    """

    if time_min > time_max:
        lag_vector = np.arange(time_max, time_min + 1)[::-1]
    else:
        lag_vector = np.arange(time_min, time_max + 1)

    return lag_vector


def quadratic_regularization(dim):
    d = 2 * np.eye(dim)
    d[[0, -1], [0, -1]] = 1

    upper = np.hstack([np.zeros((dim, 1)), np.eye(dim, dim - 1)])
    lower = np.vstack([np.zeros((1, dim)), np.eye(dim - 1, dim)])
    m = d - upper - lower

    return m


def regularized_regression_fit(X, y, m, alpha=1.0):
    """Calculate the parameters using regularized reverse correlation. Assumes,
    that a intercept term has been added manually!

    Parameters
    ----------
    X : np.ndarray
        The feature matrix. Should be in shape: times x features
    y : np.ndarray
        The target matrix. Should be in shape: times x targets
    m : np.array
        Regularization matrix, either quadratic regularization or identity matrix
    alpha : np.float
        The regularization parameter. Usually named lambda. Must be >0, defaults to
        1.0

    Returns
    ----------
    :return beta: np.array
    The beta parameters in shape target_features x features
    """

    if X.shape[0] < X.shape[1]:
        warnings.warn(f'X: more features {X.shape[1]}' +
                      f' than samples {X.shape[0]}, check input dimensions!')
    if y.shape[0] < y.shape[1]:
        warnings.warn(f'y: more features {y.shape[1]}' +
                      f' than samples {y.shape[0]}, check input dimensions!')

    assert alpha >= 0, 'reg_lambda has to be positive!'
    assert X.shape[0] == y.shape[0], f'Cannot multiply X with dim {X.shape[0]}' \
                                     + f' and y with dim {y.shape[0]}'

    if np.sum(X[:, 0] == 1) != X.shape[0]:
        warnings.warn('Please check, whether an intercept term has been added!')

    # TODO Tests: 1, 2 numerical tests

    xtx = X.T.dot(X)
    xtx += m * alpha

    xy = X.T.dot(y)

    beta = linalg.solve(xtx, xy, sym_pos=True, overwrite_a=False)

    return beta


def regularized_regression_predict(x, coefficients):
    # TODO Test output dims
    # TODO test numerical things
    # TODO assert: input dimensions and test
    y_hat = x.dot(coefficients)
    return y_hat


def model_to_coefficients(model, intercept):
    # TODO tests,
    # TODO asserts
    # Reshaping using fortran due to Matlab combability, maybe this can be dropped later.

    model = np.vstack([intercept.reshape(-1, model.shape[2]),
                       np.reshape(model, (model.shape[0] * model.shape[1],
                                          model.shape[2]), order='F')])

    return model


def coefficient_to_model(coefficients, x_dim1, n_lags, y_dim1):
    assert x_dim1 * (n_lags + 1) * y_dim1 == coefficients.ravel().shape[0], 'Check inputs!'
    intercept = coefficients[:x_dim1, :]
    # The reshape order is set to Fortran to keep Matlab compability
    model = np.reshape(coefficients[x_dim1:, :],
                       (x_dim1, n_lags, y_dim1), order='F')
    return model, intercept


def stimulus_mapping(mapping_direction, stim, resp, tmin, tmax):
    # TODO Documentation, tests
    if mapping_direction == 1:
        x = stim.copy()
        y = resp.copy()
    elif mapping_direction == -1:
        x = resp.copy()
        y = stim.copy()
        (tmin, tmax) = (tmax, tmin)
    else:
        raise ValueError('Value of mapping_direction must be 1 (forward) or -1 (backward)')

    return x, y, tmin, tmax


def test_input_dimensions(x):
    if isinstance(x, list):
        n_trials = len(x)
        n_feat = x[0].shape[1]
        for trl in x:
            assert trl.shape[-1] == n_feat, 'Number of features have to be equal!'
            assert trl.ndim == 2, 'Arrays in list have to be of size times x features'
    elif isinstance(x, np.ndarray):
        assert x.ndim == 3, 'Nd array of trials x times x features expected!'
        n_trials = x.shape[0]
        n_feat = x.shape[2]
    else:
        raise ValueError('Input shut be either a list of np.arrays or a single' +
                         ' np.array of shape trials x times x features')

    return n_trials, n_feat


def test_reg_lambda(reg_lambda):
    if isinstance(reg_lambda, (np.ndarray, list)):
        for rl in reg_lambda:
            assert rl > 0, 'Regularization has to be positive and larger than 0'
    elif isinstance(reg_lambda, float):
        assert reg_lambda > 0, 'reg_lambda has to be positive and larger than 0!'
        reg_lambda = [reg_lambda]
    else:
        raise ValueError('reg_lambda has to be a list, np.ndarray or float!')

    return reg_lambda