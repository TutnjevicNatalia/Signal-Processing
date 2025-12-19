function validate_iteration_loso()
% VALIDATE_ITERATION_LOSO
% Leave-One-Subject-Out cross-validation for Iteration 2, 3, or 4.
%
% Pipeline:
%   - load_training_data
%   - preprocess_multichannel
%   - extract_features_multichannel (iteration-dependent behaviour)
%   - (Iter 3+ only) subjectwise_zscore_features across FULL dataset
%   - For each LOSO fold:
%       * select_features on TRAIN only
%       * z-score (mu/sigma) on TRAIN only
%       * train_classifier
%       * evaluate on TEST
%
% Reports:
%   - Global pooled-epoch metrics + confusion matrix
%   - Per-subject metrics table
%   - Mean ± SD across subjects for:
%       Accuracy, Kappa, Macro-F1, Balanced Accuracy

clc;
close all;

addpath(genpath('src'));
run('config.m');

if CURRENT_ITERATION < 2 || CURRENT_ITERATION > 4
    error('validate_iteration_loso is intended for CURRENT_ITERATION in {2,3,4}.');
end

fprintf('=== LOSO validation for Iteration %d ===\n', CURRENT_ITERATION);

edf_dir = TRAINING_DIR;
xml_dir = TRAINING_DIR;

%% 1) Load full multi-channel training data
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

%% 2) Preprocess & feature extraction for ALL epochs once
fprintf('\n[1] Preprocessing multi-channel data for ALL epochs.\n');
pp_data = preprocess_multichannel(multi_channel_data, labels_all, channel_info);

fprintf('[2] Extracting multi-channel features for ALL epochs (iteration %d)...\n', CURRENT_ITERATION);
features_all = extract_features_multichannel(pp_data, channel_info, CURRENT_ITERATION);

% Iter 3+ only: subject-wise z-scoring across FULL dataset
if CURRENT_ITERATION >= 3
    fprintf('[3] Applying subject-wise z-scoring of features (Iter %d).\n', CURRENT_ITERATION);
    features_all = subjectwise_zscore_features(features_all, subject_ids);
else
    fprintf('[3] Skipping subject-wise z-scoring (Iter 2).\n');
end

%% 3) LOSO over subjects
unique_subs = unique(subject_ids);
nSubs = numel(unique_subs);

all_true = [];
all_pred = [];

acc_per_sub     = zeros(nSubs,1);
kappa_per_sub   = zeros(nSubs,1);
f1macro_per_sub = zeros(nSubs,1);
balacc_per_sub  = zeros(nSubs,1);

labels_unique = unique(labels_all);
numClasses    = numel(labels_unique);

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

    % --- Z-scoring on TRAIN only ---
    mu_fold  = mean(X_train_fs, 1);
    sig_fold = std(X_train_fs, 0, 1) + 1e-6;

    X_train = (X_train_fs - mu_fold) ./ sig_fold;
    X_test  = (X_test_fs  - mu_fold) ./ sig_fold;

    % --- Train classifier on fold ---
    model_fold = train_classifier(X_train, y_train);

    % --- Predict on test (ONLY ONCE) ---
    y_pred = predict(model_fold, X_test);
    
    if CURRENT_ITERATION >= 4
        y_pred = smooth_predictions_median(y_pred, 5);
    end


    % Collect global epoch-wise outputs
    all_true = [all_true; y_test(:)];
    all_pred = [all_pred; y_pred(:)];

    % Per-subject accuracy
    acc_per_sub(i) = mean(y_pred(:) == y_test(:));
    fprintf('    Accuracy for subject %d: %.2f %%\n', test_sub, 100*acc_per_sub(i));

    % --- Per-subject confusion + metrics ---
    C_sub = confusionmat(y_test, y_pred, 'Order', labels_unique);
    N_sub = sum(C_sub(:));

    tp = diag(C_sub);
    fn = sum(C_sub, 2) - tp;
    fp = sum(C_sub, 1)' - tp;

    precision_sub = zeros(numClasses,1);
    recall_sub    = zeros(numClasses,1);
    f1_sub        = zeros(numClasses,1);

    for k = 1:numClasses
        if tp(k) + fp(k) > 0
            precision_sub(k) = tp(k) / (tp(k) + fp(k));
        else
            precision_sub(k) = 0;
        end

        if tp(k) + fn(k) > 0
            recall_sub(k) = tp(k) / (tp(k) + fn(k));
        else
            recall_sub(k) = 0;
        end

        if precision_sub(k) + recall_sub(k) > 0
            f1_sub(k) = 2 * precision_sub(k) * recall_sub(k) / (precision_sub(k) + recall_sub(k));
        else
            f1_sub(k) = 0;
        end
    end

    f1macro_per_sub(i) = mean(f1_sub);
    balacc_per_sub(i)  = mean(recall_sub); % balanced accuracy = mean recall across classes

    % Cohen's kappa per subject
    if N_sub > 0
        p0_sub   = sum(tp) / N_sub;
        row_marg = sum(C_sub, 2);
        col_marg = sum(C_sub, 1)';
        pe_sub   = sum(row_marg .* col_marg) / (N_sub^2);

        if (1 - pe_sub) > 0
            kappa_per_sub(i) = (p0_sub - pe_sub) / (1 - pe_sub);
        else
            kappa_per_sub(i) = NaN;
        end
    else
        kappa_per_sub(i) = NaN;
    end

    % --- Optional: hypnogram plot for this subject ---
    epoch_length_sec = 30; % update if needed
    if isfield(channel_info, 'subject_names') && ~isempty(channel_info.subject_names)
        % Try to map by fold index; if not available, fall back
        if numel(channel_info.subject_names) >= i
            subj_name = channel_info.subject_names{i};
        else
            subj_name = sprintf('S%d', test_sub);
        end
    else
        subj_name = sprintf('S%d', test_sub);
    end

    try
        plot_hypnogram_comparison(y_test, y_pred, epoch_length_sec, subj_name);
    catch
        % Keep validation running even if plotting is not available
    end
end

%% 4) Overall (pooled epoch-wise) metrics
overall_acc = mean(all_true == all_pred);

fprintf('\n=== LOSO Summary (Iteration %d) ===\n', CURRENT_ITERATION);
fprintf('  Mean per-subject accuracy: %.2f %%\n', 100*mean(acc_per_sub));
fprintf('  Overall epoch-wise accuracy: %.2f %%\n', 100*overall_acc);

labels_unique_global = unique(all_true);
C = confusionmat(all_true, all_pred, 'Order', labels_unique_global);

fprintf('\nConfusion matrix (rows: true, cols: pred):\n');
disp(array2table(C, 'VariableNames', ...
    cellstr("Pred_" + string(labels_unique_global(:)')), ...
    'RowNames', cellstr("True_" + string(labels_unique_global(:)))));

%% 5) Global pooled-epoch Macro F1, balanced accuracy, Cohen's kappa
N = sum(C(:));
numClasses_global = numel(labels_unique_global);

tp = diag(C);
fn = sum(C, 2) - tp;
fp = sum(C, 1)' - tp;

precision = zeros(numClasses_global,1);
recall    = zeros(numClasses_global,1);
f1_class  = zeros(numClasses_global,1);

for k = 1:numClasses_global
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
balanced_acc = mean(recall);

p0 = sum(tp) / N;
row_marg = sum(C, 2);
col_marg = sum(C, 1)';
pe = sum(row_marg .* col_marg) / (N^2);

if (1 - pe) > 0
    kappa = (p0 - pe) / (1 - pe);
else
    kappa = NaN;
end

fprintf('\nAdditional LOSO metrics (global pooled epochs):\n');
fprintf('  Macro F1:          %.2f %%\n', 100*macro_f1);
fprintf('  Balanced accuracy: %.2f %%\n', 100*balanced_acc);
fprintf('  Cohen''s kappa:    %.3f\n', kappa);

fprintf('\nPer-class F1 scores:\n');
for k = 1:numClasses_global
    fprintf('  Class %d: F1 = %.2f %%  (precision = %.2f %%, recall = %.2f %%)\n', ...
        labels_unique_global(k), 100*f1_class(k), 100*precision(k), 100*recall(k));
end

%% 6) Per-subject table + mean ± SD across subjects
if isfield(channel_info, 'subject_names') && ~isempty(channel_info.subject_names) ...
        && numel(channel_info.subject_names) >= nSubs
    subject_labels = channel_info.subject_names(1:nSubs);
else
    subject_labels = cellstr("S" + string(unique_subs));
end

per_subject_acc_pct = 100 * acc_per_sub;
per_subject_f1_pct  = 100 * f1macro_per_sub;
per_subject_bal_pct = 100 * balacc_per_sub;

loso_table = table(subject_labels(:), per_subject_acc_pct, kappa_per_sub, per_subject_f1_pct, per_subject_bal_pct, ...
    'VariableNames', {'Subject','AccuracyPct','Kappa','F1MacroPct','BalancedAccPct'});

fprintf('\nPer-subject LOSO table (MATLAB):\n');
disp(loso_table);

mean_acc   = mean(per_subject_acc_pct);
std_acc    = std(per_subject_acc_pct, 0);

mean_kappa = mean(kappa_per_sub, 'omitnan');
std_kappa  = std(kappa_per_sub, 0, 'omitnan');

mean_f1    = mean(per_subject_f1_pct);
std_f1     = std(per_subject_f1_pct, 0);

mean_bal   = mean(per_subject_bal_pct);
std_bal    = std(per_subject_bal_pct, 0);

fprintf('\n=== Per-subject (LOSO) mean ± SD across subjects ===\n');
fprintf('  Accuracy:          %.2f ± %.2f %%\n', mean_acc, std_acc);
fprintf('  Cohen''s kappa:     %.3f ± %.3f\n', mean_kappa, std_kappa);
fprintf('  Macro F1:          %.2f ± %.2f %%\n', mean_f1, std_f1);
fprintf('  Balanced accuracy: %.2f ± %.2f %%\n', mean_bal, std_bal);

fprintf('\nPer-subject LOSO table (LaTeX):\n');
fprintf('\\textbf{Subject} & \\textbf{Accuracy (\\%%)} & \\textbf{Kappa} & \\textbf{F1-macro (\\%%)} & \\textbf{Bal. Acc (\\%%)} \\\\\n');
fprintf('\\hline\n');
for i = 1:nSubs
    fprintf('%s & %.2f & %.3f & %.2f & %.2f \\\\\n', ...
        subject_labels{i}, per_subject_acc_pct(i), kappa_per_sub(i), per_subject_f1_pct(i), per_subject_bal_pct(i));
end
fprintf('\\hline\n');
fprintf('\\textbf{Mean} & \\textbf{%.2f} & \\textbf{%.3f} & \\textbf{%.2f} & \\textbf{%.2f} \\\\\n', ...
    mean_acc, mean_kappa, mean_f1, mean_bal);
fprintf('\\textbf{Std}  & \\textbf{%.2f} & \\textbf{%.3f} & \\textbf{%.2f} & \\textbf{%.2f} \\\\\n', ...
    std_acc, std_kappa, std_f1, std_bal);
fprintf('\\hline\n\n');

end
