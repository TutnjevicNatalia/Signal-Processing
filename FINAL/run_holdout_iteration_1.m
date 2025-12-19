%% run_holdout_iter1.m
% Holdout inference for Iteration 1, matching validate_iteration1_loso()

clc; close all; clearvars;
addpath(genpath('src'));
run('config.m');

CURRENT_ITERATION = 1;
fprintf('--- Holdout inference (Iteration %d) ---\n', CURRENT_ITERATION);

%% [1] Load holdout data (your function returns multichannel struct)
[multi_hold, channel_info, record_numbers, epoch_numbers] = load_holdout_data(HOLDOUT_DIR);

if ~isfield(multi_hold,'eeg') || isempty(multi_hold.eeg)
    error('No EEG data found in holdout.');
end

fprintf('Loaded EEG: %d epochs x %d ch x %d samples/epoch\n', ...
    size(multi_hold.eeg,1), size(multi_hold.eeg,2), size(multi_hold.eeg,3));

%% [2] Combine EEG channels like Iteration 1 (mean across C3/C4)
eeg_combined = squeeze(mean(multi_hold.eeg, 2));   % -> [nEpochs x nSamples]

%% [3] Remove NaN/Inf raw epochs (2D)
bad_raw_mask = any(~isfinite(eeg_combined), 2);
fprintf('Bad raw epochs: %d\n', sum(bad_raw_mask));

if any(bad_raw_mask)
    eeg_combined(bad_raw_mask,:) = [];
    record_numbers(bad_raw_mask) = [];
    epoch_numbers(bad_raw_mask)  = [];
end

%% [4] Preprocess (Iteration 1 style) WITHOUT labels
% Your preprocess() in LOSO is called as preprocess(X, y).
% For holdout, we pass [] and handle both possible function signatures.
try
    [X_hold_pre, ~] = preprocess(eeg_combined, []);  % if preprocess expects 2 args
catch
    X_hold_pre = preprocess(eeg_combined);           % if preprocess expects 1 arg
end

%% [5] Feature extraction (Iteration 1)
feats_hold = extract_features(X_hold_pre);  % expected [nEpochs x 16]
fprintf('Features: %d epochs x %d feats\n', size(feats_hold,1), size(feats_hold,2));

%% [6] Remove NaN/Inf feature rows
bad_feat_mask = any(~isfinite(feats_hold), 2);
fprintf('Bad feature epochs: %d\n', sum(bad_feat_mask));

if any(bad_feat_mask)
    feats_hold(bad_feat_mask,:)  = [];
    record_numbers(bad_feat_mask) = [];
    epoch_numbers(bad_feat_mask)  = [];
end

%% [7] Load trained Iteration 1 model from cache
model_path = fullfile('cache','model_iter1.mat');
if ~exist(model_path,'file')
    error('Missing model file: %s', model_path);
end

S = load(model_path);

if ~isfield(S,'model')
    error('model_iter1.mat does not contain "model".');
end
model = S.model;

% Optional: feature selection + scaling (only if your training saved them)
if isfield(S,'sel_idx')
    feats_hold = feats_hold(:, S.sel_idx);
    fprintf('Applied sel_idx: now %d features\n', size(feats_hold,2));
end

if isfield(S,'mu_train') && isfield(S,'sig_train')
    feats_hold = (feats_hold - S.mu_train) ./ S.sig_train;
    fprintf('Applied training scaling (mu_train/sig_train)\n');
else
    fprintf('No mu_train/sig_train in model_iter1.mat -> no scaling applied\n');
end

%% [8] Predict
pred_labels = predict(model, feats_hold);
pred_labels = pred_labels(:);

fprintf('Predicted %d epochs\n', numel(pred_labels));

%% [9] Build 0-based epoch indices per record and write submission
file_ids = cellstr(record_numbers);
N = numel(file_ids);

epoch_idx0 = zeros(N,1);
u = unique(file_ids,'stable');
for r = 1:numel(u)
    idx = find(strcmp(file_ids,u{r}));
    epoch_idx0(idx) = 0:(numel(idx)-1);   % 0-based
end

if numel(epoch_idx0) ~= numel(pred_labels)
    error('Mismatch: epoch_idx0 (%d) vs pred_labels (%d)', numel(epoch_idx0), numel(pred_labels));
end

T = table(file_ids(:), epoch_idx0(:), pred_labels(:), ...
    'VariableNames', {'record_number','epoch_number','label'});

writetable(T, SUBMISSION_FILE);
fprintf('--- Done. Saved to %s ---\n', SUBMISSION_FILE);
