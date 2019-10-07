#import context
import pymtrf
from .simulate_test_data import build_test_data
import os
import numpy as np
from scipy.io import loadmat
t_precision = 10


def test_lag_gen_shape_neg_lags():
    fake_data = np.random.rand(20, 2)
    lags = pymtrf.lag_builder(-2, 0)
    shape = pymtrf.lag_gen(fake_data, lags).shape

    assert np.all(shape == (20, 6))


def test_lag_gen_shape_pos_lags():
    fake_data = np.random.rand(20, 2)
    lags = pymtrf.lag_builder(2, 0)
    shape = pymtrf.lag_gen(fake_data, lags).shape
    assert np.all(shape == (20, 6))


def test_lag_gen_shape_no_lags():
    fake_data = np.random.rand(20, 2)
    lags = pymtrf.lag_builder(0, 0)
    shape = pymtrf.lag_gen(fake_data, lags).shape
    assert np.all(shape == (20, 2))


def test_lag_gen_shift_pos():
    fake_data = np.ones((3, 1))
    lags = pymtrf.lag_builder(0, 1)
    lag_matrix = pymtrf.lag_gen(fake_data, lags)
    test_matrix = np.ones((3, 2))
    test_matrix[0, 1] = 0
    assert np.all(test_matrix == lag_matrix)


def test_lag_gen_shift_neg():
    fake_data = np.ones((3, 1))
    lags = pymtrf.lag_builder(-1, 0)
    lag_matrix = pymtrf.lag_gen(fake_data, lags)
    test_matrix = np.ones((3, 2))
    test_matrix[2, 0] = 0
    assert np.all(test_matrix == lag_matrix)


def test_lag_gen_shift_pos_one():
    fake_data = np.ones((3, 1))
    lags = pymtrf.lag_builder(1, 1)
    lag_matrix = pymtrf.lag_gen(fake_data, lags)
    test_matrix = np.ones((3, 1))
    test_matrix[0, 0] = 0
    assert np.all(test_matrix == lag_matrix)


def test_mtrf_train_fwd():
    data = loadmat(f'pymtrf{os.sep}test{os.sep}test_files{os.sep}mtrf_train_fwd.mat')
    w = data['w']
    t = np.squeeze(data['t'])
    i = data['i']
    x, model, y = build_test_data()
    w_t, t_t, i_t = pymtrf.mtrf_train(x[:, :3], y[:, :3], 64, 1, -60, 60, 1)
    np.testing.assert_almost_equal(w, w_t, t_precision)
    np.testing.assert_almost_equal(t, t_t, t_precision)
    np.testing.assert_almost_equal(i, i_t, t_precision)


def test_mtrf_train_bwd():
    data = loadmat(f'pymtrf{os.sep}test{os.sep}test_files{os.sep}mtrf_train_bwd.mat')
    w = data['w']
    t = np.squeeze(data['t'])
    i = data['i']
    x, _, y = build_test_data()
    w_t, t_t, i_t = pymtrf.mtrf_train(x[:, :3], y[:, :3], 64, -1, -60, 60, 1)
    np.testing.assert_almost_equal(w, w_t, t_precision)
    np.testing.assert_almost_equal(t, t_t, t_precision)
    np.testing.assert_almost_equal(i, i_t, t_precision)


def test_mtrf_predict_fwd():
    data = loadmat(f'pymtrf{os.sep}test{os.sep}test_files{os.sep}mtrf_predict_fwd.mat')
    rec = data['rec']
    r = data['r']
    p = data['p']
    mse = data['mse']
    x, _, y = build_test_data()
    w_t, t_t, i_t = pymtrf.mtrf_train(x[:64 * 2, :], y[:64*2, :4], 64, 1, 0, 60, 1)
    rec_t, r_t, p_t, mse_t = pymtrf.mtrf_predict(x[64*2:64*2*2, :],
                                                 y[64*2:64*2*2, :4], w_t, 64, 1, 0,
                                                 60, i_t)

    np.testing.assert_almost_equal(rec, rec_t, t_precision)
    np.testing.assert_almost_equal(r, r_t, t_precision)
    np.testing.assert_almost_equal(p, p_t, t_precision)
    np.testing.assert_almost_equal(mse, mse_t, t_precision)


def test_mtrf_cross_val_equal_bwd():
    x, _, y = build_test_data()
    x_train = [x[i*64*2:(i+1)*64*2, :] for i in range(4)]
    y_train = [y[i*64*2:(i+1)*64*2, :4] for i in range(4)]
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)
    data = loadmat(f'pymtrf{os.sep}test{os.sep}test_files{os.sep}cross_val_equal_bwd.mat')
    pred = data['pred']
    r = data['r']
    p = data['p']
    mse = data['mse']
    model = data['model']
    r_t, p_t, mse_t, pred_t, model_t = pymtrf.mtrf_crossval(x_train, y_train,
                                                            64, -1, -60, 60,
                                                            [0.1, 1, 10])

    np.testing.assert_almost_equal(r, r_t, t_precision)
    np.testing.assert_almost_equal(p, p_t, t_precision)
    np.testing.assert_almost_equal(mse, mse_t, t_precision)
    for (pr, pr_t) in zip(pred.T, pred_t):
        np.testing.assert_almost_equal(pr[0], pr_t, t_precision)
    np.testing.assert_almost_equal(model, model_t, t_precision)


def test_mtrf_multicross_val_equal_bwd():
    x, _, y = build_test_data()
    x_train = [x[i*64*2:(i+1)*64*2, :] for i in range(4)]
    y_train = [y[i*64*2:(i+1)*64*2, :4] for i in range(4)]
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)
    data = loadmat(f'pymtrf{os.sep}test{os.sep}test_files{os.sep}multicross_val_equal_bwd.mat')
    pred = data['pred']
    r = data['r']
    p = data['p']
    mse = data['mse']
    model = data['model']
    r_t, p_t, mse_t, pred_t, model_t = pymtrf.mtrf_multicrossval(x_train,
                                                                 y_train,
                                                                 y_train,
                                                                 y_train,
                                                                 64, -1, -60,
                                                                 60,
                                                                 [0.1, 1, 10],
                                                                 [0.1, 1, 10])

    np.testing.assert_almost_equal(r, r_t, t_precision)
    np.testing.assert_almost_equal(p, p_t, t_precision)
    np.testing.assert_almost_equal(mse, mse_t, t_precision)
    for (pr, pr_t) in zip(pred.T, pred_t):
        np.testing.assert_almost_equal(pr[0], pr_t, t_precision)
    np.testing.assert_almost_equal(model, model_t, t_precision)


def test_mtrf_cross_val_equal_fwd():
    x, _, y = build_test_data()
    x_train = [x[i*64*2:(i+1)*64*2, :] for i in range(4)]
    y_train = [y[i*64*2:(i+1)*64*2, :4] for i in range(4)]
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)
    data = loadmat(f'pymtrf{os.sep}test{os.sep}test_files{os.sep}cross_val_equal_fwd.mat')
    pred = data['pred']
    r = data['r']
    p = data['p']
    mse = data['mse']
    model = data['model']
    r_t, p_t, mse_t, pred_t, model_t = pymtrf.mtrf_crossval(x_train, y_train,
                                                            64, 1, -60, 60,
                                                            [0.1, 1, 10])

    np.testing.assert_almost_equal(r, r_t, t_precision)
    np.testing.assert_almost_equal(p, p_t, t_precision)
    np.testing.assert_almost_equal(mse, mse_t, t_precision)
    for (pr, pr_t) in zip(pred.T, pred_t):
        np.testing.assert_almost_equal(pr[0], pr_t, t_precision)
    np.testing.assert_almost_equal(model, model_t, t_precision)


def test_mtrf_multicross_val_equal_fwd():
    x, _, y = build_test_data()
    x_train = [x[i*64*2:(i+1)*64*2, :] for i in range(4)]
    y_train = [y[i*64*2:(i+1)*64*2, :4] for i in range(4)]
    x_train = np.stack(x_train)
    y_train = np.stack(y_train)
    data = loadmat(f'pymtrf{os.sep}test{os.sep}test_files{os.sep}multicross_val_equal_fwd.mat')
    pred = data['pred']
    r = data['r']
    p = data['p']
    mse = data['mse']
    model = data['model']
    r_t, p_t, mse_t, pred_t, model_t = pymtrf.mtrf_multicrossval(x_train,
                                                                 y_train,
                                                                 y_train,
                                                                 y_train,
                                                                 64, 1, -60, 60,
                                                                 [0.1, 1, 10],
                                                                 [0.1, 1, 10])

    np.testing.assert_almost_equal(r, r_t, t_precision)
    np.testing.assert_almost_equal(p, p_t, t_precision)
    np.testing.assert_almost_equal(mse, mse_t, t_precision)
    for (pr, pr_t) in zip(pred.T, pred_t):
        np.testing.assert_almost_equal(pr[0], pr_t, t_precision)
    np.testing.assert_almost_equal(model, model_t, t_precision)


def test_mtrf_cross_val_unequal_fwd():
    x, _, y = build_test_data()
    x_train = [x[i*64*2:(i+1)*64*2, :] for i in range(2)]
    y_train = [y[i*64*2:(i+1)*64*2, :4] for i in range(2)]
    x_train.append(x[1*64*2: 64*2*3, :])
    y_train.append(y[1*64*2: 64*2*3, :4])
    data = loadmat(f'pymtrf{os.sep}test{os.sep}test_files{os.sep}cross_val_unequal_fwd.mat')
    pred = data['pred']
    r = data['r']
    p = data['p']
    mse = data['mse']
    model = data['model']
    r_t, p_t, mse_t, pred_t, model_t = pymtrf.mtrf_crossval(x_train, y_train,
                                                            64, 1, -60, 60,
                                                            [0.1, 1, 10])

    np.testing.assert_almost_equal(r, r_t, t_precision)
    np.testing.assert_almost_equal(p, p_t, t_precision)
    np.testing.assert_almost_equal(mse, mse_t, t_precision)
    for (pr, pr_t) in zip(pred.T, pred_t):
        np.testing.assert_almost_equal(pr[0], pr_t, t_precision)
    np.testing.assert_almost_equal(model, model_t, t_precision)


def test_mtrf_multicross_val_unequal_fwd():
    x, _, y = build_test_data()
    x_train = [x[i*64*2:(i+1)*64*2, :] for i in range(2)]
    y_train = [y[i*64*2:(i+1)*64*2, :4] for i in range(2)]
    x_train.append(x[1*64*2: 64*2*3, :])
    y_train.append(y[1*64*2: 64*2*3, :4])
    data = loadmat(f'pymtrf{os.sep}test{os.sep}test_files{os.sep}multicross_val_unequal_fwd.mat')
    pred = data['pred']
    r = data['r']
    p = data['p']
    mse = data['mse']
    model = data['model']
    r_t, p_t, mse_t, pred_t, model_t = pymtrf.mtrf_multicrossval(x_train,
                                                                 y_train,
                                                                 y_train,
                                                                 y_train,
                                                                 64, 1, -60, 60,
                                                                 [0.1, 1, 10],
                                                                 [0.1, 1, 10])

    np.testing.assert_almost_equal(r, r_t, t_precision)
    np.testing.assert_almost_equal(p, p_t, t_precision)
    np.testing.assert_almost_equal(mse, mse_t, t_precision)
    for (pr, pr_t) in zip(pred.T, pred_t):
        np.testing.assert_almost_equal(pr[0], pr_t, t_precision)
    np.testing.assert_almost_equal(model, model_t, t_precision)


def test_mtrf_transform_fwd():
    # Output works, but matrix is quite ill conditioned (I think) therefore,
    # low precision in this step.
    t_precision = 1
    x, model, y = build_test_data()
    data = loadmat(f'pymtrf{os.sep}test{os.sep}test_files{os.sep}mtrf_transform_fwd.mat')
    model_t = data['model_t']
    t = np.squeeze(data['t'])
    c_t = data['c_t']

    [model_t_t, t_t, c_t_t] = pymtrf.mtrf_transform(x, y, model, 64, 1,
                                                    -60, 60)

    np.testing.assert_almost_equal(model_t, model_t_t, t_precision)
    np.testing.assert_almost_equal(t, t_t, t_precision)
    np.testing.assert_almost_equal(c_t, c_t_t, t_precision)


def test_mtrf_transform_bwd():
    t_precision = 10
    x, model, y = build_test_data()
    data = loadmat(f'pymtrf{os.sep}test{os.sep}test_files{os.sep}mtrf_transform_bwd.mat')
    model_t = data['model_t']
    t = np.squeeze(data['t'])
    c_t = data['c_t']

    [model_t_t, t_t, c_t_t] = pymtrf.mtrf_transform(x, y,
                                                    np.transpose(model,
                                                                 [2, 1, 0]),
                                                    64, -1, -60, 60)

    np.testing.assert_almost_equal(model_t, model_t_t, t_precision)
    np.testing.assert_almost_equal(t, t_t, t_precision)
    np.testing.assert_almost_equal(c_t, c_t_t, t_precision)

'''
Old tests based on mtrf_test_data.m, were used before for for testing of
precision etc. Now we are using simulated data for the tests. See the matlab
functions. There is some copying of files involved to get things running...
def test_mtrf_train_model_contrast():
    contr = loadmat('test_results/rdm_contrast_data')
    w = contr['w']
    t = contr['t']
    i = contr['i']
    data = loadmat('../../contrast_data.mat')
    eeg = data['EEG']
    contrast_level = data['contrastLevel']
    Fs = data['Fs']
    w_t, t_t, i_t = pymtrf.mtrf_train(contrast_level, eeg, Fs, 1, -150, 450, 1)

    np.testing.assert_almost_equal(w, w_t, decimal=10, err_msg="Failure in weights calculation")
    np.testing.assert_almost_equal(t, t_t, decimal=10, err_msg="Failure in lag calculation")
    np.testing.assert_almost_equal(i, i_t, decimal=10, err_msg="Failure in intercept calculation")

def test_mtrf_train_model_motion():
    motion = loadmat('test_results/rdm_motion_data')
    w = motion['w']
    t = motion['t']
    i = motion['i']
    data = loadmat('../../coherentmotion_data.mat')
    eeg = data['EEG']
    motion_level = data['coherentMotionLevel']
    Fs = data['Fs']
    w_t, t_t, i_t = pymtrf.mtrf_train(motion_level, eeg, Fs, 1, -150, 450, 1)

    np.testing.assert_almost_equal(w, w_t, decimal=10, err_msg="Failure in weights calculation")
    np.testing.assert_almost_equal(t, t_t, decimal=10, err_msg="Failure in lag calculation")
    np.testing.assert_almost_equal(i, i_t, decimal=10, err_msg="Failure in intercept calculation")


def test_mtrf_train_model_speech_trf():
    # TODO rename contrasts
    speech = loadmat('test_results/rdm_speech_data_trf')
    w = speech['w']
    t = speech['t']
    i = speech['i']
    data = loadmat('../../speech_data.mat')
    eeg = data['EEG']
    envelope = data['envelope']
    Fs = data['Fs']
    w_t, t_t, i_t = pymtrf.mtrf_train(envelope, eeg, Fs, 1, -150, 450, 0.1)

    np.testing.assert_almost_equal(w, w_t, decimal=10, err_msg="Failure in weights calculation")
    np.testing.assert_almost_equal(t, t_t, decimal=10, err_msg="Failure in lag calculation")
    np.testing.assert_almost_equal(i, i_t, decimal=10, err_msg="Failure in intercept calculation")


def test_mtrf_train_model_speech_strf():
    speech = loadmat('test_results/rdm_speech_data_strf')
    w = speech['w']
    t = speech['t']
    i = speech['i']
    data = loadmat('../../speech_data.mat')
    eeg = data['EEG']
    spectrogram = data['spectrogram']
    Fs = data['Fs']
    w_t, t_t, i_t = pymtrf.mtrf_train(spectrogram, eeg, Fs, 1, -150, 450, 100)

    np.testing.assert_almost_equal(w, w_t, decimal=10, err_msg="Failure in weights calculation")
    np.testing.assert_almost_equal(t, t_t, decimal=10, err_msg="Failure in lag calculation")
    np.testing.assert_almost_equal(i, i_t, decimal=10, err_msg="Failure in intercept calculation")


def test_mtrf_train_model_speech_recon_train():
    speech = loadmat('test_results/rdm_speech_data_recon')
    g = np.expand_dims(speech['g'], -1)
    t = speech['t']
    con = np.expand_dims(speech['con'], 0)
    data = loadmat('../../speech_data.mat')
    eeg = data['EEG']
    envelope = data['envelope']
    Fs = data['Fs'].astype('int')[0][0]
    eeg_train = eeg[: Fs * 60, :]

    envelope_train = envelope[: Fs * 60, :]
    g_t, t_t, con_t = pymtrf.mtrf_train(envelope_train, eeg_train, Fs, -1, 0, 500, 1e5)
    np.testing.assert_almost_equal(g, g_t, decimal=10, err_msg="Failure in weights calculation")
    np.testing.assert_almost_equal(t[0], t_t, decimal=10, err_msg="Failure in lag calculation")
    np.testing.assert_almost_equal(con[0], con_t, decimal=10, err_msg="Failure in intercept calculation")


def test_mtrf_train_model_speech_recon_predict():
    speech = loadmat('test_results/rdm_speech_data_recon')
    g = np.expand_dims(speech['g'], -1)
    con = np.expand_dims(speech['con'], 0)
    recon = speech['recon']
    r = speech['r']
    p = speech['p']
    mse = speech['MSE']
    data = loadmat('../../speech_data.mat')
    eeg = data['EEG']
    envelope = data['envelope']
    Fs = data['Fs'].astype('int')[0][0]
    eeg_test = eeg[Fs * 60:, :]
    envelope_test = envelope[Fs * 60:, :]

    recon_t, r_t, p_t, mse_t = pymtrf.mtrf_predict(envelope_test, eeg_test, g, Fs, -1, 0, 500, con)

    np.testing.assert_almost_equal(recon, recon_t, decimal=10, err_msg="Failure in reconstruction")
    np.testing.assert_almost_equal(r, r_t, decimal=10, err_msg="Failure in correlation - r")
    np.testing.assert_almost_equal(p, p_t, decimal=10, err_msg="Failure in correlation - p")
    np.testing.assert_almost_equal(mse, mse_t, decimal=10, err_msg="Failure in error calculation")


def test_mtrf_train_model_speech_cross_val_equal():
    speech = loadmat('test_results/rdm_speech_data_cross_val_equal.mat')
    model = np.expand_dims(speech['model'], -1)
    r = np.expand_dims(speech['r'], -1)
    p = np.expand_dims(speech['p'], -1)
    mse = np.expand_dims(speech['mse'], -1)
    data = loadmat('../../speech_data.mat')
    eeg = data['EEG']
    envelope = data['envelope']

    Fs = data['Fs'].astype('int')[0][0]
    eeg_in = np.stack([eeg[: Fs * 30, :], eeg[Fs * 30: Fs * 60, :], eeg[Fs * 60: Fs * 90]])
    envelope_in = np.stack([envelope[: Fs * 30, :], envelope[Fs * 30: Fs * 60, :], envelope[Fs * 60: Fs * 90]])

    r_t, p_t, mse_t, _, model_t = pymtrf.mtrf_crossval(envelope_in, eeg_in, Fs, -1, -50, 150, [0.1, 1, 10])
    np.testing.assert_almost_equal(r, r_t, decimal=9, err_msg="Failure in correlation")
    np.testing.assert_almost_equal(p, p_t, decimal=8, err_msg="Failure in p-values")
    np.testing.assert_almost_equal(mse, mse_t, decimal=8, err_msg="Failure in mse")
    np.testing.assert_almost_equal(model, model_t, decimal=8, err_msg="Failure in model")


def test_mtrf_train_model_speech_cross_val_unequal():
    speech = loadmat('test_results/rdm_speech_data_cross_val_unequal.mat')
    model = np.expand_dims(speech['model'], -1)
    r = np.expand_dims(speech['r'], -1)
    p = np.expand_dims(speech['p'], -1)
    mse = np.expand_dims(speech['mse'], -1)
    data = loadmat('../../speech_data.mat')
    eeg = data['EEG']
    envelope = data['envelope']

    Fs = data['Fs'].astype('int')[0][0]
    eeg_in = [eeg[: Fs * 30, :], eeg[Fs * 30: Fs * 60, :], eeg[Fs * 60: Fs * 100], eeg[Fs * 100:]]
    envelope_in = [envelope[: Fs * 30, :], envelope[Fs * 30: Fs * 60, :], envelope[Fs * 60: Fs * 100], envelope[Fs * 100:]]

    r_t, p_t, mse_t, _, model_t = pymtrf.mtrf_crossval(envelope_in, eeg_in, Fs, -1, -50, 150, [0.1, 1, 10])
    np.testing.assert_almost_equal(r, r_t, decimal=9, err_msg="Failure in correlation")
    np.testing.assert_almost_equal(p, p_t, decimal=8, err_msg="Failure in p-values")
    np.testing.assert_almost_equal(mse, mse_t, decimal=8, err_msg="Failure in mse")
    np.testing.assert_almost_equal(model, model_t, decimal=8, err_msg="Failure in model")


def test_mtrf_train_model_speech_multi_cross_val_equal():
    speech = loadmat('test_results/rdm_speech_data_multi_cross_val_equal.mat')
    model = np.expand_dims(speech['model'], -1)
    r = np.expand_dims(speech['r'], -1)
    p = np.expand_dims(speech['p'], -1)
    mse = np.expand_dims(speech['mse'], -1)
    data = loadmat('../../speech_data.mat')
    eeg = data['EEG']
    envelope = data['envelope']

    Fs = data['Fs'].astype('int')[0][0]
    eeg_in = np.stack([eeg[: Fs * 30, :], eeg[Fs * 30: Fs * 60, :], eeg[Fs * 60: Fs * 90]])
    envelope_in = np.stack([envelope[: Fs * 30, :], envelope[Fs * 30: Fs * 60, :], envelope[Fs * 60: Fs * 90]])

    r_t, p_t, mse_t, _, model_t = pymtrf.mtrf_multicrossval(envelope_in, eeg_in, eeg_in, eeg_in, Fs, -1, -50, 150,
                                                            [0.1, 1, 10], [0.1, 1, 10])
    np.testing.assert_almost_equal(r, r_t, decimal=9, err_msg="Failure in correlation")
    np.testing.assert_almost_equal(p, p_t, decimal=8, err_msg="Failure in p-values")
    np.testing.assert_almost_equal(mse, mse_t, decimal=8, err_msg="Failure in mse")
    np.testing.assert_almost_equal(model, model_t, decimal=8, err_msg="Failure in model")


def test_mtrf_train_model_speech_multi_cross_val_unequal():
    speech = loadmat('test_results/rdm_speech_data_multi_cross_val_unequal.mat')
    model = np.expand_dims(speech['model'], -1)
    r = np.expand_dims(speech['r'], -1)
    p = np.expand_dims(speech['p'], -1)
    mse = np.expand_dims(speech['mse'], -1)
    data = loadmat('../../speech_data.mat')
    eeg = data['EEG']
    envelope = data['envelope']

    Fs = data['Fs'].astype('int')[0][0]
    eeg_in = [eeg[: Fs * 30, :], eeg[Fs * 30: Fs * 60, :], eeg[Fs * 60: Fs * 100], eeg[Fs * 100:]]
    envelope_in = [envelope[: Fs * 30, :], envelope[Fs * 30: Fs * 60, :], envelope[Fs * 60: Fs * 100], envelope[Fs * 100:]]

    r_t, p_t, mse_t, _, model_t = pymtrf.mtrf_multicrossval(envelope_in, eeg_in, eeg_in, eeg_in, Fs, -1, -50, 150,
                                                            [0.1, 1, 10], [0.1, 1, 10])
    np.testing.assert_almost_equal(r, r_t, decimal=9, err_msg="Failure in correlation")
    np.testing.assert_almost_equal(p, p_t, decimal=8, err_msg="Failure in p-values")
    np.testing.assert_almost_equal(mse, mse_t, decimal=8, err_msg="Failure in mse")
    np.testing.assert_almost_equal(model, model_t, decimal=8, err_msg="Failure in model")


def test_mtrf_train_model_speech_cross_val_equal_fwd():
    speech = loadmat('test_results/rdm_speech_data_cross_val_equal_fwd.mat')
    model = speech['model']
    r = speech['r']
    p =speech['p']
    mse = speech['mse']
    data = loadmat('../../speech_data.mat')
    eeg = data['EEG']
    envelope = data['envelope']

    Fs = data['Fs'].astype('int')[0][0]
    eeg_in = np.stack([eeg[: Fs * 30, :], eeg[Fs * 30: Fs * 60, :], eeg[Fs * 60: Fs * 90]])
    envelope_in = np.stack([envelope[: Fs * 30, :], envelope[Fs * 30: Fs * 60, :], envelope[Fs * 60: Fs * 90]])

    r_t, p_t, mse_t, _, model_t = pymtrf.mtrf_crossval(envelope_in, eeg_in, Fs, 1, -50, 150, [0.1, 1, 10])
    np.testing.assert_almost_equal(r, r_t, decimal=9, err_msg="Failure in correlation")
    np.testing.assert_almost_equal(p, p_t, decimal=8, err_msg="Failure in p-values")
    np.testing.assert_almost_equal(mse, mse_t, decimal=8, err_msg="Failure in mse")
    np.testing.assert_almost_equal(model, model_t, decimal=8, err_msg="Failure in model")


def test_mtrf_train_model_speech_cross_val_unequal_fwd():
    speech = loadmat('test_results/rdm_speech_data_cross_val_unequal_fwd.mat')
    model = speech['model']
    r = speech['r']
    p =speech['p']
    mse = speech['mse']
    data = loadmat('../../speech_data.mat')
    eeg = data['EEG']
    envelope = data['envelope']

    Fs = data['Fs'].astype('int')[0][0]
    eeg_in = [eeg[: Fs * 30, :], eeg[Fs * 30: Fs * 60, :], eeg[Fs * 60: Fs * 100], eeg[Fs * 100:]]
    envelope_in = [envelope[: Fs * 30, :], envelope[Fs * 30: Fs * 60, :], envelope[Fs * 60: Fs * 100], envelope[Fs * 100:]]

    r_t, p_t, mse_t, _, model_t = pymtrf.mtrf_crossval(envelope_in, eeg_in, Fs, 1, -50, 150, [0.1, 1, 10])
    np.testing.assert_almost_equal(r, r_t, decimal=9, err_msg="Failure in correlation")
    np.testing.assert_almost_equal(p, p_t, decimal=9, err_msg="Failure in p-values")
    np.testing.assert_almost_equal(mse, mse_t, decimal=9, err_msg="Failure in mse")
    np.testing.assert_almost_equal(model, model_t, decimal=9, err_msg="Failure in model")


def test_mtrf_train_model_speech_multi_cross_val_equal_fwd():
    speech = loadmat('test_results/rdm_speech_data_multi_cross_val_equal_fwd.mat')
    model = speech['model']
    r = speech['r']
    p =speech['p']
    mse = speech['mse']
    data = loadmat('../../speech_data.mat')
    eeg = data['EEG']
    envelope = data['envelope']

    Fs = data['Fs'].astype('int')[0][0]
    eeg_in = np.stack([eeg[: Fs * 30, :], eeg[Fs * 30: Fs * 60, :], eeg[Fs * 60: Fs * 90]])
    envelope_in = np.stack([envelope[: Fs * 30, :], envelope[Fs * 30: Fs * 60, :], envelope[Fs * 60: Fs * 90]])

    r_t, p_t, mse_t, _, model_t = pymtrf.mtrf_multicrossval(envelope_in, eeg_in, eeg_in, eeg_in, Fs, 1, -50, 150,
                                                            [0.1, 1, 10], [0.1, 1, 10])
    np.testing.assert_almost_equal(r, r_t, decimal=9, err_msg="Failure in correlation")
    np.testing.assert_almost_equal(p, p_t, decimal=9, err_msg="Failure in p-values")
    np.testing.assert_almost_equal(mse, mse_t, decimal=9, err_msg="Failure in mse")
    np.testing.assert_almost_equal(model, model_t, decimal=9, err_msg="Failure in model")


def test_mtrf_train_model_speech_multi_cross_val_unequal_fwd():
    speech = loadmat('test_results/rdm_speech_data_multi_cross_val_unequal_fwd.mat')
    model = speech['model']
    r = speech['r']
    p =speech['p']
    mse = speech['mse']
    data = loadmat('../../speech_data.mat')
    eeg = data['EEG']
    envelope = data['envelope']

    Fs = data['Fs'].astype('int')[0][0]
    eeg_in = [eeg[: Fs * 30, :], eeg[Fs * 30: Fs * 60, :], eeg[Fs * 60: Fs * 100], eeg[Fs * 100:]]
    envelope_in = [envelope[: Fs * 30, :], envelope[Fs * 30: Fs * 60, :], envelope[Fs * 60: Fs * 100], envelope[Fs * 100:]]

    r_t, p_t, mse_t, _, model_t = pymtrf.mtrf_multicrossval(envelope_in, eeg_in, eeg_in, eeg_in, Fs, 1, -50, 150,
                                                            [0.1, 1, 10], [0.1, 1, 10])
    np.testing.assert_almost_equal(r, r_t, decimal=9, err_msg="Failure in correlation")
    np.testing.assert_almost_equal(p, p_t, decimal=9, err_msg="Failure in p-values")
    np.testing.assert_almost_equal(mse, mse_t, decimal=9, err_msg="Failure in mse")
    np.testing.assert_almost_equal(model, model_t, decimal=9, err_msg="Failure in model")
'''
