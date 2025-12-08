%% run_holdout.m
% Run inference on HOLDOUT data (H1..H10) with:
%   - NaN/Inf checks & removal on raw data
%   - NaN/Inf checks & removal on features
%   - Remove 1 extra epoch per record (to match expected counts: 10411)
%   - Subject-wise z-score then global scaling from training
%   - 0-based epoch indexing in submission (0..N-1)
%
% Uses:
%   - Trained model + feature selection + scaling from cache/model_iterX.mat
%   - NO retraining inside this script.

clc;
close all;
clearvars;

addpath(genpath('src'));

% Load configuration
run('config.m');

if CURRENT_ITERATION < 2
    error('run_holdout.m is written for multi-channel (Iteration >= 2). Set CURRENT_ITERATION >= 2 in config.m.');
end

fprintf('--- Inference on Holdout Data (Iteration %d) ---\n', CURRENT_ITERATION);
fprintf('Using HOLDOUT_DIR = %s\n', HOLDOUT_DIR);

edf_holdout_dir = HOLDOUT_DIR;

%% [1] Load holdout data (multi-channel, NO labels/XML)
fprintf('\n[1] Loading holdout data...\n');

[multi_channel_holdout, channel_info_holdout, record_numbers, epoch_numbers] = ...
    load_holdout_data(edf_holdout_dir);

n_epochs_eeg = size(multi_channel_holdout.eeg, 1);
n_ids        = numel(record_numbers);

fprintf('Finished building HOLDOUT data.\n');
fprintf('  EEG: %d epochs, %d channels, %d samples/epoch\n', ...
    size(multi_channel_holdout.eeg,1), size(multi_channel_holdout.eeg,2), size(multi_channel_holdout.eeg,3));
if isfield(multi_channel_holdout,'eog')
    fprintf('  EOG: %d epochs, %d channels, %d samples/epoch\n', ...
        size(multi_channel_holdout.eog,1), size(multi_channel_holdout.eog,2), size(multi_channel_holdout.eog,3));
end
if isfield(multi_channel_holdout,'emg')
    fprintf('  EMG: %d epochs, %d channels, %d samples/epoch\n', ...
        size(multi_channel_holdout.emg,1), size(multi_channel_holdout.emg,2), size(multi_channel_holdout.emg,3));
end

fprintf('Sanity check: EEG epochs = %d, record_numbers = %d\n', n_epochs_eeg, n_ids);
if n_epochs_eeg ~= n_ids
    warning('Mismatch between EEG epochs (%d) and record_numbers (%d)!', n_epochs_eeg, n_ids);
end

%% [2] Check raw data for NaN/Inf and remove bad epochs
fprintf('\n[2] Checking raw multi-channel data for NaN/Inf...\n');

[bad_raw_mask, bad_raw_per_rec] = find_bad_epochs_raw(multi_channel_holdout, record_numbers);

fprintf('  Bad raw epochs (NaN/Inf): %d\n', sum(bad_raw_mask));

if sum(bad_raw_mask) > 0
    fprintf('  Bad raw epochs by record:\n');
    rec_names = fieldnames(bad_raw_per_rec);
    for i = 1:numel(rec_names)
        fprintf('    %s: %d epochs\n', rec_names{i}, bad_raw_per_rec.(rec_names{i}));
    end

    multi_channel_holdout = remove_epochs_from_multichan(multi_channel_holdout, bad_raw_mask);

    record_numbers(bad_raw_mask) = [];
    epoch_numbers(bad_raw_mask)  = [];
    if isfield(channel_info_holdout, 'subject_ids') && numel(channel_info_holdout.subject_ids) == numel(bad_raw_mask)
        channel_info_holdout.subject_ids(bad_raw_mask) = [];
    end

    fprintf('  After removing raw-bad epochs, total EEG epochs = %d\n', size(multi_channel_holdout.eeg,1));
else
    fprintf('  No NaN/Inf found in raw signals.\n');
end

%% [3] Preprocess holdout data (multi-channel, no labels)
fprintf('\n[3] Preprocessing holdout data (multichannel)...\n');

[preprocessed_holdout, ~] = preprocess_multichannel(multi_channel_holdout, [], channel_info_holdout);

%% [4] Extract features for holdout data
fprintf('\n[4] Extracting features for holdout data...\n');

features_hold_raw = extract_features_multichannel(preprocessed_holdout, ...
                                                  channel_info_holdout, CURRENT_ITERATION);

fprintf('  Feature matrix size: %d epochs x %d features\n', ...
        size(features_hold_raw,1), size(features_hold_raw,2));

if size(features_hold_raw,1) ~= numel(record_numbers)
    warning('Features have %d epochs but record_numbers has %d.', ...
            size(features_hold_raw,1), numel(record_numbers));
end

%% [5] Check features for NaN/Inf and remove bad epochs
fprintf('\n[5] Checking feature matrix for NaN/Inf...\n');

[bad_feat_mask, bad_feat_per_rec] = find_bad_epochs_features(features_hold_raw, record_numbers);
fprintf('  Bad feature epochs (NaN/Inf): %d\n', sum(bad_feat_mask));

if sum(bad_feat_mask) > 0
    fprintf('  Bad feature epochs by record:\n');
    rec_names = fieldnames(bad_feat_per_rec);
    for i = 1:numel(rec_names)
        fprintf('    %s: %d epochs\n', rec_names{i}, bad_feat_per_rec.(rec_names{i}));
    end

    features_hold_raw(bad_feat_mask, :) = [];
    record_numbers(bad_feat_mask)       = [];
    epoch_numbers(bad_feat_mask)        = [];
    if isfield(channel_info_holdout, 'subject_ids') && numel(channel_info_holdout.subject_ids) == numel(bad_feat_mask)
        channel_info_holdout.subject_ids(bad_feat_mask) = [];
    end

    fprintf('  After removing feature-bad epochs, total epochs = %d\n', size(features_hold_raw,1));
else
    fprintf('  No NaN/Inf found in features.\n');
end

%% [6] Remove the extra epoch per record (to match expected counts)
fprintf('\n[6] Removing 1 extra epoch per record (drop LAST epoch of each H*)...\n');

file_ids = cellstr(record_numbers);            % e.g. 'H1','H1',...,'H10',...
unique_recs = unique(file_ids, 'stable');

remove_mask_extra = false(size(file_ids));

for r = 1:numel(unique_recs)
    idx = find(strcmp(file_ids, unique_recs{r}));
    if ~isempty(idx)
        remove_mask_extra(idx(end)) = true;    % drop last epoch of each H*
    end
end

fprintf('  Total epochs before removing extra: %d\n', size(features_hold_raw,1));
fprintf('  Removing %d epochs (one per recording).\n', sum(remove_mask_extra));

features_hold_raw(remove_mask_extra, :) = [];
record_numbers(remove_mask_extra)       = [];
epoch_numbers(remove_mask_extra)        = [];
if isfield(channel_info_holdout, 'subject_ids') && numel(channel_info_holdout.subject_ids) == numel(remove_mask_extra)
    channel_info_holdout.subject_ids(remove_mask_extra) = [];
end

fprintf('  Total epochs after removing extra: %d\n', size(features_hold_raw,1));

%% [7] Load trained model + feature selection from cache
fprintf('\n[7] Loading trained model and feature selection info from cache...\n');

model_filename = sprintf('model_iter%d.mat', CURRENT_ITERATION);
model_path     = fullfile(CACHE_DIR, model_filename);

if ~exist(model_path, 'file')
    error('Trained model file not found: %s\nRun main.m on training data first.', model_path);
end

S = load(model_path);  % expects: model, sel_idx, mu_train, sig_train

if ~isfield(S, 'model') || ~isfield(S, 'sel_idx') ...
        || ~isfield(S, 'mu_train') || ~isfield(S, 'sig_train')
    error('Cache file %s does not contain model+sel_idx+mu_train+sig_train. Re-run main.m.', model_path);
end

model    = S.model;
sel_idx  = S.sel_idx;
mu_train = S.mu_train;
sig_train = S.sig_train;

fprintf('  Loaded model and feature selection info from %s\n', model_path);

%% [8] Subject-wise normalisation, feature selection & scaling
fprintf('\n[8] Applying subject-wise normalisation, feature selection and scaling to holdout features...\n');

% Subject-wise z-scoring for holdout subjects (Iter 3+ only)
if CURRENT_ITERATION >= 3 && isfield(channel_info_holdout, 'subject_ids')
    fprintf('  Subject-wise z-scoring of HOLDOUT features (Iter 3+).\n');
    features_hold_raw = subjectwise_zscore_features(features_hold_raw, ...
                                                    channel_info_holdout.subject_ids);
end

% Apply SAME feature subset as in training
features_hold_fs = features_hold_raw(:, sel_idx);
fprintf('  Holdout feature matrix after selection: %d epochs x %d features\n', ...
        size(features_hold_fs,1), size(features_hold_fs,2));

% Scale using TRAINING means/std
X_hold = (features_hold_fs - mu_train) ./ sig_train;

%% [9] Predict sleep stages for holdout data
fprintf('\n[9] Predicting sleep stages for holdout data...\n');

pred_labels = predict(model, X_hold);
pred_labels = pred_labels(:);

fprintf('  Predictions computed: %d epochs.\n', numel(pred_labels));

%% [10] Build 0-based epoch indices per record and write submission
fprintf('\n[10] Building 0-based epoch indices per recording and writing submission...\n');

file_ids = cellstr(record_numbers);   % ensure cellstr
N = numel(file_ids);

epoch_idx = zeros(N,1);
unique_recs = unique(file_ids, 'stable');

for r = 1:numel(unique_recs)
    idx = find(strcmp(file_ids, unique_recs{r}));
    epoch_idx(idx) = 0:(numel(idx)-1);   % 0-based indexing
end

if numel(epoch_idx) ~= numel(pred_labels)
    error('Size mismatch: epoch_idx (%d) vs pred_labels (%d).', ...
          numel(epoch_idx), numel(pred_labels));
end

T = table(file_ids(:), epoch_idx(:), pred_labels(:), ...
          'VariableNames', {'record_number','epoch_number','label'});

writetable(T, SUBMISSION_FILE);
fprintf('--- Holdout inference finished. Submission saved to %s ---\n', SUBMISSION_FILE);



%% ===== Local helper functions =====

function [bad_mask, per_rec] = find_bad_epochs_raw(multi_channel_data, record_numbers)
    n_epochs = size(multi_channel_data.eeg,1);
    bad_mask = false(n_epochs,1);

    if isfield(multi_channel_data, 'eeg') && ~isempty(multi_channel_data.eeg)
        bad_mask = bad_mask | mark_bad_epochs_3d(multi_channel_data.eeg);
    end
    if isfield(multi_channel_data, 'eog') && ~isempty(multi_channel_data.eog)
        bad_mask = bad_mask | mark_bad_epochs_3d(multi_channel_data.eog);
    end
    if isfield(multi_channel_data, 'emg') && ~isempty(multi_channel_data.emg)
        bad_mask = bad_mask | mark_bad_epochs_3d(multi_channel_data.emg);
    end

    per_rec = count_bad_by_record(bad_mask, record_numbers);
end

function [bad_mask, per_rec] = find_bad_epochs_features(features, record_numbers)
    bad_mask = any(~isfinite(features), 2);
    per_rec  = count_bad_by_record(bad_mask, record_numbers);
end

function bad = mark_bad_epochs_3d(data)
    if isempty(data)
        bad = false(0,1);
        return;
    end
    bad = squeeze(any(any(~isfinite(data), 3), 2));
    bad = reshape(bad, [], 1);
end

function per_rec = count_bad_by_record(bad_mask, record_numbers)
    per_rec = struct();
    idx_bad = find(bad_mask);
    if isempty(idx_bad)
        return;
    end
    bad_files = record_numbers(idx_bad);
    u = unique(bad_files, 'stable');
    for i = 1:numel(u)
        key = char(u(i));
        per_rec.(key) = sum(bad_files == u(i));
    end
end

function mc = remove_epochs_from_multichan(mc, mask)
    fields = fieldnames(mc);
    for i = 1:numel(fields)
        f = fields{i};
        if ~isempty(mc.(f))
            mc.(f)(mask,:,:) = [];
        end
    end
end
