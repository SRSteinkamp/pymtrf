import numpy as np
from scipy.stats import pearsonr
from .helper import *
import warnings


def lag_gen(data, time_lags):
    '''lag_gen returns the matrix containing the lagged time series of data for
    a range of time lags given by the list or numpy array lags. If the data is
    multivariate, lag_gen concatenates the features for each lag along the
    columns of the output array.

    Parameters
    ----------
    data : {float, array_like}, shape = [n_samples, n_features]
        The training data, i.e. the data that is shifted in time.
    time_lags : {int, array_like}, shape = [n_lags]
        Indices for lags that will be applied to the data.

    Returns
    -------
    lagged_data : {float, array_like}, shape = [n_samples, n_features * n_lag}
        The data shifted in time, as described above.

    See also
    --------
    mtrf_train : calculate forward or backward models.
    mtrf_predict : predict stimulus or response based on models.
    mtrf_crossval : calculate reconstruction accuracies for a dataset.

    Translation to Python: Simon Richard Steinkamp
    Github:
    October 2018; Last revision: 18.January 2019
    Original MATLAB toolbox, mTRF v. 1.5
    Author: Michael Crosse
    Lalor Lab, Trinity College Dublin, IRELAND
    Email: edmundlalor@gmail.com
    Website: http://lalorlab.net/
    April 2014; Last revision: 18 August 2015
    '''
    lagged_data = np.zeros((data.shape[0], data.shape[1] * time_lags.shape[0]))

    chan = 0
    for lags in time_lags:

        if lags < 0:
            lagged_data[:lags, chan:chan + data.shape[1]] = data[-lags:, :]
        elif lags > 0:
            lagged_data[lags:, chan:chan + data.shape[1]] = data[:-lags, :]
        else:
            lagged_data[:, chan:chan + data.shape[1]] = data

        chan = chan + data.shape[1]

    return lagged_data


def mtrf_train(stim, resp, fs, mapping_direction, tmin, tmax, reg_lambda):
    '''performs ridge regression on the stimulus property stim and the neural
    response data resp to solve for their linear mapping function. Pass in
    mapping_direction = 1 to map in the forward direction or -1 to map
    backwards. The sampling frequency fs should be defined in Hertz and the
    time lags should be set in milliseconds between tmin and tmap.
    Regularisation is controlled by the ridge parameter reg_lambda.

    Parameters
    ----------
    stim : {float, array_like}, shape = [n_samples, n_features]
        The stimulus property.
    resp : {float, array_like}, shape = [n_samples, n_channels]
        The neural data.
    fs : {float}
        sampling frequency
    mapping_direction : {1, -1}
        mapping direction
    tmin : {float}
        minimum time lag in ms
    tmax : {float}
        maximum time lag in ms
    reg_lambda : {float}
        ridge parameter

    Returns
    -------
    model : {float, array_like}, shape = [n_features, n_lags, n_targets]
        linear mapping function, features by lags by channels for
        mapping_direction = 1, channels by lags by features for
        mapping_direction = -1.
    time_lags : {array_like}, shape = [n_lags]
        vector of time lags in ms
    intercept : {array_like}, shape = [n_features]
        the regression constant.
    See also
    --------
    mtrf_train : calculate forward or backward models.
    mtrf_predict : predict stimulus or response based on models.
    mtrf_crossval : calculate reconstruction accuracies for a dataset.

    References
    ----------
    [1] Lalor EC, Pearlmutter BA, Reilly RB, McDarby G, Foxe JJ (2006)
        The VESPA: a method for the rapid estimation of a visual evoked
        potential. NeuroImage 32:1549-1561.
    [1] Crosse MC, Di Liberto GM, Bednar A, Lalor EC (2015) The
        multivariate temporal response function (mTRF) toolbox: a MATLAB
        toolbox for relating neural signals to continuous stimuli. Front
        Hum Neurosci 10:604.

    Translation to Python: Simon Richard Steinkamp
    Github:
    October 2018; Last revision: 18.January 2019
    Original MATLAB toolbox, mTRF v. 1.5
    Author: Edmund Lalor, Michael Crosse, Giovanni Di Liberto
    Lalor Lab, Trinity College Dublin, IRELAND
    Email: edmundlalor@gmail.com
    Website: http://lalorlab.net/
    April 2014; Last revision: 8 January 2016
    '''
    if stim.shape[0] < stim.shape[1]:
        warnings.warn(f'stim: more features {stim.shape[0]} ' +
                      f'than samples {stim.shape[1]}, check input dimensions!')
    if resp.shape[0] < resp.shape[1]:
        warnings.warn(f'resp: more features {resp.shape[0]} ' +
                      f'than samples {resp.shape[1]}, check input dimensions!')

    assert tmin < tmax, 'tmin has to be smaller than tmax'
    assert reg_lambda > 0, 'reg_lambda has to be positive and larger than 0!'

    x, y, tmin, tmax = stimulus_mapping(mapping_direction, stim, resp, tmin, tmax)

    t_min = np.floor(tmin / 1e3 * fs * mapping_direction).astype(int)
    t_max = np.ceil(tmax / 1e3 * fs * mapping_direction).astype(int)

    lags = lag_builder(t_min, t_max)
    lag_x = lag_gen(x, lags)
    x_input = np.hstack([np.ones(x.shape), lag_x])
    n_feat = x_input.shape[1]

    if x.shape[1] == 1:
        reg_matrix = quadratic_regularization(n_feat)
    else:
        reg_matrix = np.eye(n_feat)

    coefficients = regularized_regression_fit(x_input, y, reg_matrix, reg_lambda)
    model, intercept = coefficient_to_model(coefficients, x.shape[1],
                                            lags.shape[0], y.shape[1])
    time_lags = lags / fs * 1e3

    return model, time_lags, intercept


def mtrf_predict(stim, resp, model, fs, mapping_direction, tmin, tmax, constant):
    '''performs a convolution of the stimulus property or the neural response
    data with their linear mapping function (estimated by mtrf_train) to solve
    predict the neural response (mapping_direction = 1) or the stimulus property
    (mapping_direction = 1).

    Parameters
    ----------
    stim : {float, array_like}, shape = [n_samples, n_features]
        The stimulus property.
    resp : {float, array_like}, shape = [n_samples, n_channels]
        The neural data.
    model : {float, array_like}, shape = [ ]
        linear mapping function, features by lags by channels for
        mapping_direction = 1, channels by lags by features for
        mapping_direction = -1.
    fs : {float}
        sampling frequency in hz
    mapping_direction : {1, -1}
        mapping direction for forward or backward modeling
    tmin : {float}
        minimum time lag in ms
    tmax : {float}
        maximum time lag in ms
    constant : {float, array_like}
        Regression constant, if None is given, a zero constant is assumed.

    Returns
    -------
    pred : {float, array_like}, shape = [n_times, n_features]
        prediction of the regression model
    r : {float}
        correlation coefficient between prediction and original data
    p : {float}
        p-value corresponding to r
    mse : {float}
        mean squared error of prediction

    See also
    --------
    mtrf_train : calculate forward or backward models.
    mtrf_predict : predict stimulus or response based on models.
    mtrf_crossval : calculate reconstruction accuracies for a dataset.

    Translation to Python: Simon Richard Steinkamp
    Github:
    October 2018; Last revision: 18.January 2019
    Original MATLAB toolbox, mTRF v. 1.5
    Author: Michael Crosse, Giovanni Di Liberto
    Lalor Lab, Trinity College Dublin, IRELAND
    Email: edmundlalor@gmail.com
    Website: http://lalorlab.net/
    April 2014; Last revision: 8 January 2016
    '''

    # Define x and y
    assert tmin < tmax, 'Value of tmin must be < tmax'

    if constant is None:
        constant = np.zeros((model.shape[0], model.shape[2]))
    else:
        assert np.all(constant.shape == np.array([model.shape[0],
                                                 model.shape[2]]))

    x, y, tmin, tmax = stimulus_mapping(mapping_direction, stim, resp, tmin, tmax)

    t_min = np.floor(tmin / 1e3 * fs * mapping_direction).astype(int)
    t_max = np.ceil(tmax / 1e3 * fs * mapping_direction).astype(int)

    lags = lag_builder(t_min, t_max)

    x_lag = np.hstack([np.ones(x.shape), lag_gen(x, lags)])

    model = model_to_coefficients(model, constant)

    pred = regularized_regression_predict(x_lag, model)

    # Calculate accuracy
    if y is not None:
        r = np.zeros((1, y.shape[1]))
        p = np.zeros((1, y.shape[1]))
        mse = np.zeros((1, y.shape[1]))
        for i in range(y.shape[1]):
            r[:, i], p[:, i] = pearsonr(y[:, i], pred[:, i])
            mse[:, i] = np.mean((y[:, i] - pred[:, i]) ** 2)
    else:
        r = None
        p = None
        mse = None

    return pred, r, p, mse


def mtrf_crossval(stim, resp, fs, mapping_direction, tmin, tmax, reg_lambda):
    '''performs leave-one-out cross-validation on the set of stimuli and the
    neural responses for a range of ridge parameter values.
    As a measure of performance, the correlation coefficients between the
    predicted and original signals, the corresponding p-values, and the mean
    squared errors (mse) are returned. Forward and backward modelling can be
    performed.

    Parameters
    ----------
    stim : {float, array_like}, shape = [n_trials, n_samples, n_features]
        The stimulus property, can be an array including the different trials
        in the first dimension or a list of arrays of
        shape = [n_sample, n_features].
    resp : {float, array_like}, shape = [n_trials, n_samples, n_features]
        The neural data, can be an array including the different trials
        in the first dimension or a list of arrays of
        shape = [n_sample, n_features].
    fs : {float}
        sampling frequency
    mapping_direction : {1, -1}
        mapping direction, 1 for forward mapping, -1 for backward mapping
    tmin : {float}
        minimum time lag in ms
    tmax : {float}
        maximum time lag in ms
    reg_lambda : {float}
        list of ridge parameters

    Returns
    -------
    r : {float, np.ndarray}, shape = [n_trials, n_lambdas]
        correlation coefficient between prediction and original data for each
        trial and each parameter.
    p : {float, np.ndarray}, shape = [n_trials, n_lambdas]
        p-values corresponding to r
    mse : {float, np.ndarray}, shape = [n_trials, n_lambdas]
        mean squared error of the predictions
    pred : {list}
        list of length n_trials, containing np.ndarrays with shape =
        [n_lambdas, n_timepoints, n_targets]
    model : {float, np.ndarray}, shape = [n_trials, n_lambdas,
            n_lags, n_targets, n_features]
        linear mapping function from resp to stim or vice versa, depending on the
        mapping direction.

    References
    ----------
       [1] Crosse MC, Di Liberto GM, Bednar A, Lalor EC (2015) The
           multivariate temporal response function (mTRF) toolbox: a MATLAB
           toolbox for relating neural signals to continuous stimuli. Front
           Hum Neurosci 10:604.

    Translation to Python: Simon Richard Steinkamp
    Github:
    October 2018; Last revision: 18.January 2019
    Original MATLAB toolbox, mTRF v. 1.5
    Author: Michael Crosse
    Lalor Lab, Trinity College Dublin, IRELAND
    Email: edmundlalor@gmail.com
    Website: http://lalorlab.net/
    April 2014; Last revision: 31 May 2016
    '''
    assert tmin < tmax, 'Value of tmin must be < tmax'

    x, y, tmin, tmax = stimulus_mapping(mapping_direction, stim, resp, tmin, tmax)

    n_trials, n_feat = test_input_dimensions(x)

    n_trl_y, n_targets = test_input_dimensions(y)

    assert n_trials == n_trl_y, 'stim and resp should have the same no of trials!'

    reg_lambda = test_reg_lambda(reg_lambda)

    n_lambda = len(reg_lambda)

    t_min = np.floor(tmin / 1e3 * fs * mapping_direction).astype(int)
    t_max = np.ceil(tmax / 1e3 * fs * mapping_direction).astype(int)

    lags = lag_builder(t_min, t_max)

    # Set up regularisation
    dim1 = n_feat * lags.shape[0] + n_feat
    model = np.zeros((n_trials, n_lambda, dim1, n_targets))

    if n_feat == 1:
        reg_matrix = quadratic_regularization(dim1)
    else:
        reg_matrix = np.eye(dim1)

    # Training
    x_input = []

    for c_trials in range(n_trials):
        # Generate lag matrix
        x_input.append(np.hstack([np.ones(x[c_trials].shape), lag_gen(x[c_trials], lags)]))
        # Calculate model for each lambda value
        for c_lambda in range(n_lambda):
            temp = regularized_regression_fit(x_input[c_trials],
                                              y[c_trials], reg_matrix, reg_lambda[c_lambda])
            model[c_trials, c_lambda, :, :] = temp

    r = np.zeros((n_trials, n_lambda, n_targets))
    p = np.zeros(r.shape)
    mse = np.zeros(r.shape)
    pred = []

    for trial in range(n_trials):
        pred.append(np.zeros((n_lambda, y[trial].shape[0], n_targets)))

        # Perform cross-validation for each lambda value
        for c_lambda in range(n_lambda):
            # Calculate prediction
            cv_coef = np.mean(model[np.arange(n_trials) != trial, c_lambda, :, :], 0, keepdims=False)
            pred[trial][c_lambda, :, :] = regularized_regression_predict(x_input[trial], cv_coef)

            # Calculate accuracy
            for k in range(n_targets):
                temp_pred = np.squeeze(pred[trial][c_lambda, :, k]).T
                r[trial, c_lambda, k], p[trial, c_lambda, k] = pearsonr(y[trial][:, k], temp_pred)
                mse[trial, c_lambda, k] = np.mean((y[trial][:, k] - temp_pred) ** 2)

    return r, p, mse, pred, model


def mtrf_transform(stim, resp, model, fs, mapping_direction, tmin, tmax, constant=None):
    '''transforms the coefficients of the model weights into transformed model
    coefficients.
    Parameters
    ----------
    stim : {float, array_like}, shape = [n_samples, n_features]
        The stimulus property.
    resp : {float, array_like}, shape = [n_samples, n_channels]
        The neural data.
    model : {float, array_like}, shape = [ ]
        linear mapping function, features by lags by channels for
        mapping_direction = 1, channels by lags by features for
        mapping_direction = -1.
    fs : {float}
        sampling frequency in hz
    mapping_direction : {1, -1}
        mapping direction for forward or backward modeling
    tmin : {float}
        minimum time lag in ms
    tmax : {float}
        maximum time lag in ms
    intercept : {float, array_like}
        Regression constant.

    Returns
    -------
    model_t : {float, array_like}, shape = [n_times, n_features]
        transformed model weights
    t : {float}, shape = [n_lags]
        vector of time lags used in ms
    intercept_t : {float}
        transformed model constant

    See also
    --------
    mtrf_train : calculate forward or backward models.
    mtrf_predict : predict stimulus or response based on models.
    mtrf_crossval : calculate reconstruction accuracies for a dataset.

    References
    ----------
    [1] Haufe S, Meinecke F, Gorgen K, Dahne S, Haynes JD, Blankertz B,
        Bießmann F (2014) On the interpretation of weight vectors of
        linear models in multivariate neuroimaging. NeuroImage 87:96-110.

    Translation to Python: Simon Richard Steinkamp
    Github:
    October 2018; Last revision: 18.January 2019

    Original MATLAB toolbox, mTRF v. 1.5
    Author: Adam Bednar, Emily Teoh, Giovanni Di Liberto, Michael Crosse
    Lalor Lab, Trinity College Dublin, IRELAND
    Email: edmundlalor@gmail.com
    Website: http://lalorlab.net/
    April 2016; Last revision: 15 July 2016
    '''
    # Define x and y
    assert tmin < tmax, 'Value of tmin must be < tmax'

    x, y, tmin, tmax = stimulus_mapping(mapping_direction, stim, resp, tmin, tmax)

    if constant is None:
        constant = np.zeros((model.shape[0], model.shape[2]))
    else:
        assert np.all(constant.shape == np.array([model.shape[0],
                                                 model.shape[2]]))

    t_min = np.floor(tmin / 1e3 * fs * mapping_direction).astype(int)
    t_max = np.ceil(tmax / 1e3 * fs * mapping_direction).astype(int)

    lags = lag_builder(t_min, t_max)

    X = np.hstack([np.ones(x.shape), lag_gen(x, lags)])

    # Transform model weights
    model = model_to_coefficients(model, constant)
    coef_t = (X.T.dot(X)).dot(model).dot(np.linalg.inv((y.T.dot(y))))

    model_t, constant_t = coefficient_to_model(coef_t, x.shape[1], lags.shape[0], y.shape[1])
    t = lags / fs * 1e3

    return model_t, t, constant_t


def mtrf_multicrossval(stim, resp, resp1, resp2, fs, mapping_direction, tmin, tmax, reg_lambda1, reg_lambda2):
    '''performs leave-one-out cross-validation of an additive model for a
    multisensory dataset as follows:
    1. Separate unisensory models are calculated using the set of stimuli
        properties and unisensory neural responses (resp1, resp2) for each
        of their respective ridge parameters (reg_lambda1, reg_lambda2)
    2. The algebraic sums of the unisensory models for every combination of
        ridge parameters are calculated, i.e. the additive models.
    3. The additive models are validated by testing them on the set of
        multisensory neural responses.
    As a measure of performance, the correlation coefficients between the
    predicted and original signals, the corresponding p-values, and the mean
    squared errors (mse) are returned. Forward and backward modelling can be
    performed.

    Parameters
    ----------
    stim : {float, array_like}, shape = [n_trials, n_samples, n_features]
        The stimulus property, can be an array including the different trials
        in the first dimension or a list of arrays of
        shape = [n_sample, n_features].
    resp : {float, array_like}, shape = [n_trials, n_samples, n_features]
        The multisensory neural responses, can be an array including the different trials
        in the first dimension or a list of arrays of
        shape = [n_sample, n_features].
    resp1 : {float, array_like}, shape = [n_trials, n_samples, n_features]
        The first set of unisensory neural responses, can be an array including
        the different trials in the first dimension or a list of arrays of
        shape = [n_sample, n_features].
    resp2 : {float, array_like}, shape = [n_trials, n_samples, n_features]
        The first set of unisensory neural responses, can be an array including
        the different trials in the first dimension or a list of arrays of
        shape = [n_sample, n_features].
    fs : {float}
        sampling frequency
    mapping_direction : {1, -1}
        mapping direction, 1 for forward mapping, -1 for backward mapping
    tmin : {float}
        minimum time lag in ms
    tmax : {float}
        maximum time lag in ms
    reg_lambda1 : {float}
        list of ridge parameters for resp1
    reg_lambda2 : {float}
        list of ridge parameters for resp2

    Returns
    -------
    r : {float, np.ndarray}, shape = [n_trials, n_lambdas1, n_lambdas2]
        correlation coefficient between prediction and original data for each
        trial and each parameter.
    p : {float, np.ndarray}, shape = [n_trials, n_lambdas1, n_lambdas2]
        p-values corresponding to r
    mse : {float, np.ndarray}, shape = [n_trials, n_lambdas1, n_lambdas2]
        mean squared error of the predictions
    pred : {list}
        list of length n_trials, containing np.ndarrays with shape =
        [n_lambdas1, n_lambdas2, n_timepoints, n_targets]
    model : {float, np.ndarray}, shape = [n_trials, n_lambdas1, n_lambdas2,
            n_lags, n_targets, n_features]
        linear mapping function from resp to stim or vice versa, depending on the
        mapping direction.

    References
    ----------
    [1] Crosse MC, Butler JS, Lalor EC (2015) Congruent visual speech
        enhances cortical entrainment to continuous auditory speech in
        noise-free conditions. J Neurosci 35(42):14195-14204.

    Translation to Python: Simon Richard Steinkamp
    Github:
    October 2018; Last revision: 09.January 2019
    Original MATLAB toolbox, mTRF v. 1.5
    Author: Michael Crosse
    Lalor Lab, Trinity College Dublin, IRELAND
    Email: edmundlalor@gmail.com
    Website: http://lalorlab.net/
    April 2014; Last revision: 13 December 2016
    '''

    assert tmin < tmax, 'Value of tmin must be < tmax'

    x, y, tmin, tmax = stimulus_mapping(mapping_direction, stim, resp, tmin, tmax)

    reg_lambda1 = test_reg_lambda(reg_lambda1)
    reg_lambda2 = test_reg_lambda(reg_lambda2)
    n_lambda1 = len(reg_lambda1)
    n_lambda2 = len(reg_lambda2)
    t_min = np.floor(tmin / 1e3 * fs * mapping_direction).astype(int)
    t_max = np.ceil(tmax / 1e3 * fs * mapping_direction).astype(int)

    lags = lag_builder(t_min, t_max)
    n_trials, n_feat = test_input_dimensions(x)

    n_trl_y, n_targets = test_input_dimensions(y)

    assert n_trials == n_trl_y, 'stim and resp should have the same no of trials!'

    dim1 = n_feat * lags.shape[0] + n_feat

    model = np.zeros((n_trials, n_lambda1, n_lambda2, dim1, n_targets))

    if n_feat == 1:
        reg_matrix = quadratic_regularization(dim1)
    else:
        reg_matrix = np.eye(dim1)

    x_input = []

    for trial in range(n_trials):
        # Generate lag matrix
        x_input.append(np.hstack([np.ones(x[trial].shape), lag_gen(x[trial], lags)]))
        model_1_temp = np.zeros((n_lambda1, dim1, n_targets))
        model_2_temp = np.zeros((n_lambda2, dim1, n_targets))
        if mapping_direction == 1:
            # calculating unisensory model for each lambda
            # Calculate model for each lambda value
            for c_lambda, alpha in enumerate(reg_lambda1):
                temp = regularized_regression_fit(x_input[trial],
                                                  resp1[trial], reg_matrix, alpha)
                model_1_temp[c_lambda, :, :] = temp

            for c_lambda, alpha in enumerate(reg_lambda2):
                temp = regularized_regression_fit(x_input[trial],
                                                  resp2[trial], reg_matrix, alpha)
                model_2_temp[c_lambda, :, :] = temp

        elif mapping_direction == -1:
            resp1_lag = np.hstack([np.ones(resp1[trial].shape),
                                   lag_gen(resp1[trial], lags)])
            resp2_lag = np.hstack([np.ones(resp2[trial].shape),
                                   lag_gen(resp2[trial], lags)])
            for c_lambda in range(n_lambda1):
                temp = regularized_regression_fit(resp1_lag, y[trial],
                                                  reg_matrix, reg_lambda1[c_lambda])
                model_1_temp[c_lambda, :, :] = temp

            for c_lambda in range(n_lambda2):
                temp = regularized_regression_fit(resp2_lag, y[trial],
                                                  reg_matrix, reg_lambda2[c_lambda])
                model_2_temp[c_lambda, :, :] = temp

        for c_lambda1 in range(n_lambda1):
            for c_lambda2 in range(n_lambda2):
                model[trial, c_lambda1, c_lambda2, :, :] = (model_1_temp[c_lambda1, :, :]
                                                            + model_2_temp[c_lambda2, :, :])

    r = np.zeros((n_trials, n_lambda1, n_lambda2, n_targets))
    p = np.zeros(r.shape)
    mse = np.zeros(r.shape)
    pred = []

    for trial in range(n_trials):
        pred.append(np.zeros((n_lambda1, n_lambda2, y[trial].shape[0], n_targets)))

        for c_lambda1 in range(n_lambda1):
            for c_lambda2 in range(n_lambda2):
                # Calculate prediction
                cv_coef = np.mean(model[np.arange(n_trials) != trial, c_lambda1, c_lambda2, :, :], 0, keepdims=False)
                pred[trial][c_lambda1, c_lambda2, :, :] = regularized_regression_predict(x_input[trial], cv_coef)
                # Calculate accuracy
                for c_target in range(n_targets):
                    temp_pred = np.squeeze(pred[trial][c_lambda1, c_lambda2, :, c_target]).T

                    (r[trial, c_lambda1, c_lambda2, c_target],
                     p[trial, c_lambda1, c_lambda2, c_target]) = pearsonr(y[trial][:, c_target], temp_pred)
                    mse[trial, c_lambda1, c_lambda2, c_target] = np.mean((y[trial][:, c_target] - temp_pred) ** 2)

    return r, p, mse, pred, model
