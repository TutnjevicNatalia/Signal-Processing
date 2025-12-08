function validate_iteration3_loso()
% VALIDATE_ITERATION3_LOSO
% Leave-One-Subject-Out cross-validation for Iteration 3.
%
% Uses:
%   - load_training_data
%   - preprocess_multichannel
%   - extract_features_multichannel (with Iter 3 behaviour)
%   - subjectwise_zscore_features
%   - select_features
%   - train_classifier
%
% Each fold:
%   - Leave one subject out as test
%   - Train on the rest
%   - Apply feature selection & scaling on training only
%   - Evaluate on test

clc;
close all;

addpath(genpath('src'));

run('config.m');

if CURRENT_ITERATION < 3
    error('validate_iteration3_loso is intended for CURRENT_ITERATION >= 3');
end

fprintf('=== LOSO validation for Iteration %d ===\n', CURRENT_ITERATION);

edf_dir = TRAINING_DIR;
xml_dir = TRAINING_DIR;

%% 1. Load full multi-channel training data
[multi_channel_data, labels, channel_info] = load_training_data(edf_dir, xml_dir);

fprintf('Multi-channel data loaded for LOSO:\n');
if isfield(multi_channel_data,'eeg')
    fprintf('  EEG: %d epochs, %d channels, %d samples/epoch\n', ...
        size(multi_channel_data.eeg,1), size(multi_channel_data.eeg,2), size(multi_channel_data.eeg,3));
end
if isfield(multi_channel_data,'eog')
    fprintf('  EOG: %d epochs, %d channels, %d samples/epoch\n', ...
        size(multi_channel_data.eog,1), size(multi_channel_data.eog,2), size(multi_channel_data.eog,3));
end
if isfield(multi_channel_data,'emg')
    fprintf('  EMG: %d epochs, %d channels, %d samples/epoch\n', ...
        size(multi_channel_data.emg,1), size(multi_channel_data.emg,2), size(multi_channel_data.emg,3));
end

if ~isfield(channel_info, 'subject_ids') || isempty(channel_info.subject_ids)
    error('channel_info.subject_ids is required for LOSO but is missing or empty.');
end

subject_ids = channel_info.subject_ids(:);
labels_all  = labels(:);

%% 2. Preprocess & feature extraction for ALL epochs once
fprintf('\n[1] Preprocessing multi-channel data for ALL epochs.\n');
pp_data = preprocess_multichannel(multi_channel_data, labels_all, channel_info);

fprintf('[2] Extracting multi-channel features for ALL epochs (iteration %d)...\n', CURRENT_ITERATION);
features_all = extract_features_multichannel(pp_data, channel_info, CURRENT_ITERATION);

% Subject-wise normalisation across FULL dataset (Iter 3+)
fprintf('[3] Applying subject-wise z-scoring of features (Iter 3+).\n');
features_all = subjectwise_zscore_features(features_all, subject_ids);

%% 3. LOSO over subjects
unique_subs = unique(subject_ids);
nSubs = numel(unique_subs);

all_true = [];
all_pred = [];
acc_per_sub = zeros(nSubs,1);

fprintf('\n[4] Starting LOSO loop over %d subjects.\n', nSubs);

for i = 1:nSubs
    test_sub = unique_subs(i);
    fprintf('\n  Fold %d / %d: test subject = %d\n', i, nSubs, test_sub);

    test_idx  = (subject_ids == test_sub);
    train_idx = ~test_idx;

    X_train_raw = features_all(train_idx, :);
    X_test_raw  = features_all(test_idx, :);

    y_train = labels_all(train_idx);
    y_test  = labels_all(test_idx);

    % --- Feature selection on TRAIN only ---
    [X_train_fs, sel_idx] = select_features(X_train_raw, y_train);
    X_test_fs = X_test_raw(:, sel_idx);

    % --- Global z-scoring on TRAIN only ---
    mu_fold  = mean(X_train_fs, 1);
    sig_fold = std(X_train_fs, 0, 1) + 1e-6;

    X_train = (X_train_fs - mu_fold) ./ sig_fold;
    X_test  = (X_test_fs  - mu_fold) ./ sig_fold;

    % --- Train classifier on fold ---
    model_fold = train_classifier(X_train, y_train);

    % --- Predict on test ---
    y_pred = predict(model_fold, X_test);

    all_true = [all_true; y_test(:)];
    all_pred = [all_pred; y_pred(:)];

    % Per-subject accuracy
    acc_per_sub(i) = mean(y_pred(:) == y_test(:));
    fprintf('    Accuracy for subject %d: %.2f %%\n', test_sub, 100*acc_per_sub(i));
end

%% 4. Overall metrics
overall_acc = mean(all_true == all_pred);
fprintf('\n=== LOSO Summary (Iteration %d) ===\n', CURRENT_ITERATION);
fprintf('  Mean per-subject accuracy: %.2f %%\n', 100*mean(acc_per_sub));
fprintf('  Overall epoch-wise accuracy: %.2f %%\n', 100*overall_acc);

% Confusion matrix
labels_unique = unique(all_true);
C = confusionmat(all_true, all_pred, 'Order', labels_unique);

fprintf('\nConfusion matrix (rows: true, cols: pred):\n');
disp(array2table(C, 'VariableNames', ...
    cellstr("Pred_" + string(labels_unique(:)')), ...
    'RowNames', cellstr("True_" + string(labels_unique(:)))));

%% 5. Macro F1, balanced accuracy, Cohen''s kappa

N = sum(C(:));
numClasses = numel(labels_unique);

tp = diag(C);                        % true positives per class
fn = sum(C, 2) - tp;                 % row sums minus tp  (false negatives)
fp = sum(C, 1)' - tp;                % col sums minus tp  (false positives)
tn = N - tp - fn - fp;               % not used for F1, but for completeness

% Per-class precision, recall, F1
precision = zeros(numClasses,1);
recall    = zeros(numClasses,1);
f1_class  = zeros(numClasses,1);

for k = 1:numClasses
    if tp(k) + fp(k) > 0
        precision(k) = tp(k) / (tp(k) + fp(k));
    else
        precision(k) = 0;
    end

    if tp(k) + fn(k) > 0
        recall(k) = tp(k) / (tp(k) + fn(k));
    else
        recall(k) = 0;
    end

    if precision(k) + recall(k) > 0
        f1_class(k) = 2 * precision(k) * recall(k) / (precision(k) + recall(k));
    else
        f1_class(k) = 0;
    end
end

macro_f1 = mean(f1_class);

% Balanced accuracy = mean per-class recall
balanced_acc = mean(recall);

% Cohen's kappa
p0 = sum(tp) / N;                           % observed agreement
row_marg = sum(C, 2);                       % true counts per class
col_marg = sum(C, 1)';                      % predicted counts per class
pe = sum(row_marg .* col_marg) / (N^2);     % expected agreement by chance

if (1 - pe) > 0
    kappa = (p0 - pe) / (1 - pe);
else
    kappa = NaN;
end

fprintf('\nAdditional LOSO metrics:\n');
fprintf('  Macro F1:          %.2f %%\n', 100*macro_f1);
fprintf('  Balanced accuracy: %.2f %%\n', 100*balanced_acc);
fprintf('  Cohen''s kappa:    %.3f\n', kappa);

% Optional: print per-class F1
fprintf('\nPer-class F1 scores:\n');
for k = 1:numClasses
    fprintf('  Class %d: F1 = %.2f %%  (precision = %.2f %%, recall = %.2f %%)\n', ...
        labels_unique(k), 100*f1_class(k), 100*precision(k), 100*recall(k));
end

end
