% Creates matlab tests sets for the toolbox
MTRF_DIR = ''; % Change accordingly;
OUT_DIR = ['test_results' filesep]; % Change accordingly (pymtrf/tests/test_results)
mkdir([OUT_DIR])
addpath(genpath(MTRF_DIR))

% Readme set 1: 
load([MTRF_DIR 'contrast_data.mat']);
[w, t, i] = mTRFtrain(contrastLevel, EEG, Fs, 1, -150, 450, 1);
save([OUT_DIR 'rdm_contrast_data.mat'], 'w', 't', 'i')
clear Fs EEG w t i contrastLevel

% Readme set 2:
load([MTRF_DIR 'coherentmotion_data.mat']);
[w, t, i] = mTRFtrain(coherentMotionLevel, EEG, Fs, 1, -150, 450, 1);
save([OUT_DIR 'rdm_motion_data.mat'], 'w', 't', 'i')
clear Fs EEG w t i coherentMotionLevel
%%
% Readme set 3:
load([MTRF_DIR 'speech_data.mat']);
[w, t, i] = mTRFtrain(envelope, EEG, 128, 1, -150, 450, 0.1);
save([OUT_DIR 'rdm_speech_data_trf.mat'], 'w', 't', 'i')
clear w t i 

% Readme set 4
[w, t, i] = mTRFtrain(spectrogram, EEG, 128, 1, -150, 450, 100);
save([OUT_DIR 'rdm_speech_data_strf.mat'], 'w', 't', 'i')
clear w t i 

% Readme set 5:without resampling because of toolbox requirement
stimTrain = envelope(1:Fs*60,1); 
respTrain = EEG(1:Fs*60,:);
stimTest = envelope(Fs*60 + 1:end, 1);
respTest = EEG(Fs*60 + 1:end, :);
tic;
[g, t, con] = mTRFtrain(stimTrain, respTrain, Fs, -1, 0, 500, 1e5);
disp(toc)
[recon,r,p,MSE] = mTRFpredict(stimTest, respTest, g, Fs, -1, 0, 500, con);
save([OUT_DIR 'rdm_speech_data_recon.mat'], 'g', 't', 'con', 'recon', 'r', 'p', 'MSE')

clear r p MSE w t i g con recon
% Not example: Crossvall predict, unequal
stim1 = envelope(1:Fs*30,1);
stim2 = envelope(Fs*30 +1 : Fs * 60, 1);
stim3 = envelope(Fs * 60 + 1:Fs*100, 1);
stim4 = envelope(Fs * 100 + 1: end, 1);
resp1 = EEG(1:Fs*30,:);
resp2 = EEG(Fs*30 +1 : Fs * 60, :);
resp3 = EEG(Fs * 60 + 1:Fs*100, :);
resp4 = EEG(Fs * 100 + 1: end, :);

[r,p,mse,pred,model] = mTRFcrossval({stim1, stim2, stim3, stim4}, {resp1, resp2, resp3, resp4}, Fs, -1, -50, 150, [0.1, 1, 10]);
save([OUT_DIR 'rdm_speech_data_cross_val_unequal.mat'], 'r', 'p', 'mse', 'pred', 'model')

[r,p,mse,pred,model] = mTRFmulticrossval({stim1, stim2, stim3, stim4}, {resp1, resp2, resp3, resp4}, {resp1, resp2, resp3, resp4},{resp1, resp2, resp3, resp4}, Fs, -1, -50, 150, [0.1, 1, 10], [0.1, 1, 10]);
save([OUT_DIR 'rdm_speech_data_multi_cross_val_unequal.mat'], 'r', 'p', 'mse', 'pred', 'model')

clear 'r' 'p' 'mse' 'pred' 'model' 'stim1' 'stim2' 'stim3' 'stim4' 'resp1' 'resp2' 'resp3' 'resp4'
stim1 = envelope(1:Fs*30,1);
stim2 = envelope(Fs*30 +1 : Fs * 60, 1);
stim3 = envelope(Fs * 60 + 1:Fs*90, 1);
%stim4 = envelope(Fs * 90 + 1: end, 1);
resp1 = EEG(1:Fs*30,:);
resp2 = EEG(Fs*30 +1 : Fs * 60, :);
resp3 = EEG(Fs * 60 + 1:Fs*90, :);
%resp4 = EEG(Fs * 90 + 1: end, :);

[r,p,mse,pred,model] = mTRFcrossval({stim1, stim2, stim3}, {resp1, resp2, resp3}, Fs, -1, -50, 150, [0.1, 1, 10]);
save([OUT_DIR 'rdm_speech_data_cross_val_equal.mat'], 'r', 'p', 'mse', 'pred', 'model')

[r,p,mse,pred,model] = mTRFmulticrossval({stim1, stim2, stim3}, {resp1, resp2, resp3}, {resp1, resp2, resp3}, {resp1, resp2, resp3}, Fs, -1, -50, 150, [0.1, 1, 10], [0.1, 1, 10]);
save([OUT_DIR 'rdm_speech_data_multi_cross_val_equal.mat'], 'r', 'p', 'mse', 'pred', 'model')

% Not example: Crossvall predict, unequal
stim1 = envelope(1:Fs*30,1);
stim2 = envelope(Fs*30 +1 : Fs * 60, 1);
stim3 = envelope(Fs * 60 + 1:Fs*100, 1);
stim4 = envelope(Fs * 100 + 1: end, 1);
resp1 = EEG(1:Fs*30,:);
resp2 = EEG(Fs*30 +1 : Fs * 60, :);
resp3 = EEG(Fs * 60 + 1:Fs*100, :);
resp4 = EEG(Fs * 100 + 1: end, :);

[r,p,mse,pred,model] = mTRFcrossval({stim1, stim2, stim3, stim4}, {resp1, resp2, resp3, resp4}, Fs, 1, -50, 150, [0.1, 1, 10]);
save([OUT_DIR 'rdm_speech_data_cross_val_unequal_fwd.mat'], 'r', 'p', 'mse', 'pred', 'model')
%%
[r,p,mse,pred,model] = mTRFmulticrossval({stim1, stim2, stim3, stim4}, {resp1, resp2, resp3, resp4}, {resp1, resp2, resp3, resp4}, {resp1, resp2, resp3, resp4}, Fs, 1, -50, 150, [0.1, 1, 10], [0.1, 1, 10]);
save([OUT_DIR 'rdm_speech_data_multi_cross_val_unequal_fwd.mat'], 'r', 'p', 'mse', 'pred', 'model')

clear 'r' 'p' 'mse' 'pred' 'model' 'stim1' 'stim2' 'stim3' 'stim4' 'resp1' 'resp2' 'resp3' 'resp4'
stim1 = envelope(1:Fs*30,1);
stim2 = envelope(Fs*30 +1 : Fs * 60, 1);
stim3 = envelope(Fs * 60 + 1:Fs*90, 1);
%stim4 = envelope(Fs * 90 + 1: end, 1);
resp1 = EEG(1:Fs*30,:);
resp2 = EEG(Fs*30 +1 : Fs * 60, :);
resp3 = EEG(Fs * 60 + 1:Fs*90, :);
%resp4 = EEG(Fs * 90 + 1: end, :);

[r,p,mse,pred,model] = mTRFcrossval({stim1, stim2, stim3}, {resp1, resp2, resp3}, Fs, 1, -50, 150, [0.1, 1, 10]);
save([OUT_DIR 'rdm_speech_data_cross_val_equal_fwd.mat'], 'r', 'p', 'mse', 'pred', 'model')

[r,p,mse,pred,model] = mTRFmulticrossval({stim1, stim2, stim3}, {resp1, resp2, resp3}, {resp1, resp2, resp3}, {resp1, resp2, resp3}, Fs, 1, -50, 150, [0.1, 1, 10], [0.1, 1, 10]);
save([OUT_DIR 'rdm_speech_data_multi_cross_val_equal_fwd.mat'], 'r', 'p', 'mse', 'pred', 'model')