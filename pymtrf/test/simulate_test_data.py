import numpy as np
from scipy.stats import norm
from pymtrf.helper import lag_builder, model_to_coefficients
from pymtrf.helper import regularized_regression_predict
from pymtrf.mtrf import lag_gen
from scipy.io import savemat

def build_test_data(save_to_file=False, noise=1e-5):
    # Model: we define 10 channels, 9 lags, 6 targets.
    # model is channel by lags by target.
    np.random.seed(221)
    model = np.zeros((5, 9, 6))

    for i in range(6):
        model[0, :, i] = np.sin(np.linspace(0, (1 + (i/10)) * np.pi, 9))
        model[1, :, i] = 0.5 * np.sin(np.linspace(0, (1 + (i/10)) * np.pi, 9))
        model[2, :, i] = np.cos(np.linspace(0, (1 + (i/10)) * np.pi, 9))
        model[3, :, i] = 0.5 * np.cos(np.linspace(0, (1 + (i/10)) * np.pi, 9))
        model[4, :, i] = norm.pdf(np.linspace(-1, 1, 9), scale=1 + (i/10))
        # model[5, :, i] = norm.pdf(np.linspace(-1, 1, 9), loc=0 + (i/10))

    fs = 64
    tmin = -60
    tmax = 60
    mapping_direction = 1
    t_min = np.floor(tmin / 1e3 * fs * mapping_direction).astype(int)
    t_max = np.ceil(tmax / 1e3 * fs * mapping_direction).astype(int)

    lags = lag_builder(t_min, t_max)

    x = np.random.rand(8 * fs, 5)
    x = x + np.random.randn(x.shape[0], x.shape[1]) * noise
    x_lag = lag_gen(x, lags)
    x_lag = np.hstack([np.ones(x.shape), x_lag])

    coef = model_to_coefficients(model[:, :, :], np.zeros((5, 6)))

    y_sim = regularized_regression_predict(x_lag, coef)

    if save_to_file:
        savemat('gendata.mat', {'x': x, 'model': model, 'y_sim': y_sim})

    return x, model, y_sim
