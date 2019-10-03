% Script to create matlab test sets for the mtrf toolbox
% These can be seen as a precision test of the python translation
% versus the original toolbox.
% The tests have been performed using simulated data already using matlab9.1
% (2016b)

MTRF_DIR = ''; % Change towards the directory where the mtrf toolbox is located
OUT_DIR = ['test_files' filesep]; % Directory where test files are stored.
TEST_DATA = []; % Path to the simulated data.
mkdir([OUT_DIR]) % Create result folder
addpath(genpath([MTRF_DIR])) % Put mtrf toolbox on path

Fs = 64; % Set during simulation
tmin = -60; % ms from simulation
tmax = 60; % ms form simulation

SIMDATA = load([TEST_DATA filesep 'gendata.mat']);


x = SIMDATA.x; % x has shape (8 * 64, 5)
y = SIMDATA.y_sim; % y has shape (8 * 64, 6)
model = SIMDATA.model;

constant = zeros(size(model, 1), size(model, 3));

[w, t, i] = mTRFtrain(x(:,1:3), y(:,1:3), Fs, 1, -60, 60, 1);
save([OUT_DIR filesep 'mtrf_train_fwd.mat'], 'w', 't', 'i')
clear w t i

[w, t, i] = mTRFtrain(x(:,1:3), y(:,1:3), Fs, -1, -60, 60, 1);
save([OUT_DIR filesep 'mtrf_train_bwd.mat'], 'w', 't', 'i')
clear w t i

% Create train splits:
train_x = {};
train_y = {};
for i = 1 : 4
    train_x{i} = x( (i-1) * (Fs*2) + 1  : i * (Fs*2),1:5);
    train_y{i} = y( (i-1) * (Fs*2) + 1  : i * (Fs*2),1:4);
end;

[w, t, i] = mTRFtrain(train_x{1}, train_y{1}, Fs, 1, 0, 60, 1);
[rec, r, p, mse] = mTRFpredict(train_x{2}, train_y{2}, w, Fs, 1, 0, 60, i);
save([OUT_DIR filesep 'mtrf_predict_fwd.mat'], 'rec', 'r', 'p', 'mse')

clear w t i rec r p mse

[r,p,mse,pred,model] = mTRFcrossval(train_x, train_y, Fs, -1, -60, 60, [0.1, 1, 10]);
save([OUT_DIR 'cross_val_equal_bwd.mat'], 'r', 'p', 'mse', 'pred', 'model')
clear  r p mse mse pred model

[r,p,mse,pred,model] = mTRFmulticrossval(train_x, train_y, train_y, train_y, Fs, -1, -60, 60, [0.1, 1, 10], [0.1, 1, 10]);
save([OUT_DIR 'multicross_val_equal_bwd.mat'], 'r', 'p', 'mse', 'pred', 'model')

clear  r p mse mse pred model

[r,p,mse,pred,model] = mTRFcrossval(train_x, train_y, Fs, 1, -60, 60, [0.1, 1, 10]);
save([OUT_DIR 'cross_val_equal_fwd.mat'], 'r', 'p', 'mse', 'pred', 'model')
clear  r p mse mse pred model

[r,p,mse,pred,model] = mTRFmulticrossval(train_x, train_y, train_y, train_y, Fs, 1, -60, 60, [0.1, 1, 10], [0.1, 1, 10]);
save([OUT_DIR 'multicross_val_equal_fwd.mat'], 'r', 'p', 'mse', 'pred', 'model')
clear  r p mse mse pred model train_x train_y

train_x = {};
train_y = {};
for i = 1 : 2
    train_x{i} = x( (i-1) * (Fs*2) + 1  : i * (Fs*2),1:5);
    train_y{i} = y( (i-1) * (Fs*2) + 1  : i * (Fs*2),1:4);
end;

train_x{i+1} = x( (i-1) * (Fs*2) + 1  : i * (Fs*3),1:5);
train_y{i+1} = y( (i-1) * (Fs*2) + 1  : i * (Fs*3),1:4);

[r,p,mse,pred,model] = mTRFcrossval(train_x, train_y, Fs, 1, -60, 60, [0.1, 1, 10]);
save([OUT_DIR 'cross_val_unequal_fwd.mat'], 'r', 'p', 'mse', 'pred', 'model')
clear  r p mse mse pred model

[r,p,mse,pred,model] = mTRFmulticrossval(train_x, train_y, train_y, train_y, Fs, 1, -60, 60, [0.1, 1, 10], [0.1, 1, 10]);
save([OUT_DIR 'multicross_val_unequal_fwd.mat'], 'r', 'p', 'mse', 'pred', 'model')
clear  r p mse mse pred train_x train_y
%%
sim_shape = size(SIMDATA.model);
[model_t,t,c_t] = mTRFtransform(x, y, SIMDATA.model,  Fs, 1, -60, 60, zeros(sim_shape(1),sim_shape(3)));
save([OUT_DIR 'mtrf_transform_fwd.mat'], 'model_t', 't', 'c_t')
clear  model_t t c_t

[model_t,t,c_t] = mTRFtransform(x, y, permute(SIMDATA.model, [3,2,1]),  Fs, -1, -60, 60, zeros(sim_shape(3),sim_shape(1)));
save([OUT_DIR 'mtrf_transform_bwd.mat'], 'model_t', 't', 'c_t')
clear  model_t t c_t