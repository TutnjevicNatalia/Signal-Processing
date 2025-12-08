function run_submission()
%% RUN_SUBMISSION  Train on R1..R10, predict on H1..H* and write submission.csv

clc; close all;
addpath(genpath('src'));

% Load configuration (defines CURRENT_ITERATION, DATA_DIR, TRAINING_DIR, HOLDOUT_DIR, SUBMISSION_FILE, etc.)
run('config.m');

if CURRENT_ITERATION < 2
    error('This run_submission assumes multi-channel pipeline (Iteration 2+). Set CURRENT_ITERATION >= 2 in config.m.');
end

fprintf('=== Running submission pipeline (Iteration %d) ===\n', CURRENT_ITERATION);

edf_dir     = TRAINING_DIR;
xml_dir     = TRAINING_DIR;
holdout_dir = HOLDOUT_DIR;

%% 1) Load labeled training data (multi-channel)
[multi_channel_data_train, labels_train, channel_info_train] = load_training_data(edf_dir, xml_dir);
labels_train = labels_train(:);

%% 2) Preprocess + extract features for training
fprintf('\n--- Preprocessing TRAIN data ---\n');
pp_train = preprocess_multichannel(multi_channel_data_train, labels_train, channel_info_train);

fprintf('--- Extracting features for TRAIN data ---\n');
features_train_raw = extract_features_multichannel(pp_train, channel_info_train, CURRENT_ITERATION);

%% 3) Feature selection on TRAIN only
[features_train_fs, sel_idx] = select_features(features_train_raw, labels_train);

%% 4) Standardize (z-score) on TRAIN only
mu_train = mean(features_train_fs, 1);
sig_train = std(features_train_fs, 0, 1) + 1e-6;

X_train_sc = (features_train_fs - mu_train) ./ sig_train;

%% 5) Train classifier on all training subjects
fprintf('\n--- Training final model on ALL training data ---\n');
model = train_classifier(X_train_sc, labels_train);

%% 6) Load HOLDOUT data
fprintf('\n=== Loading HOLDOUT data ===\n');
[multi_channel_data_holdout, channel_info_holdout, record_numbers, epoch_numbers] = load_holdout_data(holdout_dir);

nHoldoutEpochs = size(multi_channel_data_holdout.eeg,1);
dummy_labels = zeros(nHoldoutEpochs,1);  % not used by preprocess_multichannel, but required by signature

%% 7) Preprocess + extract features for HOLDOUT
fprintf('\n--- Preprocessing HOLDOUT data ---\n');
pp_hold = preprocess_multichannel(multi_channel_data_holdout, dummy_labels, channel_info_holdout);

fprintf('--- Extracting features for HOLDOUT data ---\n');
features_hold_raw = extract_features_multichannel(pp_hold, channel_info_holdout, CURRENT_ITERATION);

%% 8) Apply SAME feature selection + scaling
X_hold_fs = features_hold_raw(:, sel_idx);
X_hold_sc = (X_hold_fs - mu_train) ./ sig_train;

%% 9) Predict labels for HOLDOUT
fprintf('\n--- Predicting on HOLDOUT epochs ---\n');
predictions = predict(model, X_hold_sc);
predictions = predictions(:);

%% 10) Write submission.csv
% Wrap config into a struct for inference_generate_submission_file
conf_struct = struct();
conf_struct.DATA_DIR        = DATA_DIR;
conf_struct.SUBMISSION_FILE = SUBMISSION_FILE;

inference_generate_submission_file(predictions, record_numbers, epoch_numbers, conf_struct);

fprintf('\nDone! Submission saved to: %s\n', fullfile(DATA_DIR, SUBMISSION_FILE));
end
