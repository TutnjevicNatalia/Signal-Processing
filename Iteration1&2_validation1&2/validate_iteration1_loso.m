function results = validate_iteration1_loso()
%% VALIDATE_ITERATION1_LOSO
% Leave-One-Subject-Out (LOSO) cross-validation for Iteration 1.
%
% Uses:
%   - Combined EEG (C3 & C4 averaged)
%   - Iteration 1 preprocessing (single-channel style)
%   - 16 time-domain features
%   - k-NN classifier (k from config.m)
%
% Outputs:
%   results struct with:
%     .fold_acc       : [10 x 1] per-subject accuracy
%     .mean_acc       : scalar
%     .std_acc        : scalar
%     .conf_mat       : [5 x 5] aggregated confusion matrix
%     .subject_names  : {'R1', ..., 'R10'}

    clc; close all;

    % --- Load config and force Iteration 1 settings ---
    run('config.m');   % loads CURRENT_ITERATION etc into base workspace

    % We want Iteration 1 behavior: kNN, time-domain features, single-channel
    assignin('base','CURRENT_ITERATION', 1);
    assignin('base','CLASSIFIER_TYPE', 'knn');
    % (KNN_N_NEIGHBORS is already set in config for iteration 1)

    % Disable cache to avoid mixing things across folds
    assignin('base','USE_CACHE', false);

    global TRAINING_DIR;
    if isempty(TRAINING_DIR)
        error('TRAINING_DIR not found. Make sure config.m defines it.');
    end

    % --- Load multi-channel data so we can get per-epoch subject IDs ---
    [mc_data, labels, channel_info] = load_training_data(TRAINING_DIR, TRAINING_DIR);

    subject_ids   = channel_info.subject_ids(:);    % [nEpochs x 1], values 1..10
    subject_names = channel_info.subject_names;     % {'R1','R2',...,'R10'}

    % Combine EEG channels (C3 & C4) into one signal as in Iteration 1
    % mc_data.eeg is [epochs x 2 x samples]
    eeg_combined = squeeze(mean(mc_data.eeg, 2));   % [nEpochs x nSamples]

    nSubjects = numel(unique(subject_ids));
    if nSubjects ~= 10
        warning('Expected 10 subjects, found %d', nSubjects);
    end

    nClasses = 5;   % 0..4 -> Wake, N1, N2, N3, REM

    fold_acc = zeros(nSubjects, 1);
    conf_mat = zeros(nClasses, nClasses);   % rows=true, cols=pred

    % For reporting
    per_subject_counts = zeros(nSubjects, 1);

    fprintf('\n=== Iteration 1 LOSO Cross-Validation ===\n');

    for s = 1:nSubjects
        fprintf('\n--- Fold %d / %d: leaving out subject %s ---\n', ...
                s, nSubjects, subject_names{s});

        test_idx  = (subject_ids == s);
        train_idx = ~test_idx;

        X_train_raw = eeg_combined(train_idx, :);
        y_train     = labels(train_idx);

        X_test_raw  = eeg_combined(test_idx, :);
        y_test      = labels(test_idx);

        per_subject_counts(s) = numel(y_test);

        % --- Preprocessing (Iteration 1 style) ---
        % We re-use your single-channel preprocess function.
        % It should behave the same as in main for iteration 1.
        [X_train_pre, y_train_pp] = preprocess(X_train_raw, y_train);
        [X_test_pre,  y_test_pp]  = preprocess(X_test_raw,  y_test);

        % Sanity: shapes should match
        if size(X_train_pre,1) ~= numel(y_train_pp)
            error('Train preprocess mismatch: %d epochs vs %d labels', ...
                  size(X_train_pre,1), numel(y_train_pp));
        end
        if size(X_test_pre,1) ~= numel(y_test_pp)
            error('Test preprocess mismatch: %d epochs vs %d labels', ...
                  size(X_test_pre,1), numel(y_test_pp));
        end

        % --- Feature extraction (16 time-domain features) ---
        feats_train = extract_features(X_train_pre);  % [nTrain x 16]
        feats_test  = extract_features(X_test_pre);   % [nTest x 16]

        % --- Train classifier (Iteration 1 = kNN) ---
        model = train_classifier(feats_train, y_train_pp);

        % --- Evaluate on held-out subject ---
        y_pred = predict(model, feats_test);
        y_pred = y_pred(:);
        y_true = y_test_pp(:);

        acc_fold = mean(y_pred == y_true) * 100;
        fold_acc(s) = acc_fold;

        fprintf('  Subject %s accuracy: %.2f%% (n = %d epochs)\n', ...
                subject_names{s}, acc_fold, numel(y_true));

        % --- Update confusion matrix (labels are 0..4) ---
        % rows: true, cols: predicted
        for i = 1:numel(y_true)
            r = y_true(i) + 1;   % 0->1,1->2,...,4->5
            c = y_pred(i) + 1;
            conf_mat(r, c) = conf_mat(r, c) + 1;
        end
    end

    % --- Overall stats ---
    mean_acc = mean(fold_acc);
    std_acc  = std(fold_acc);

    fprintf('\n=== LOSO Results (Iteration 1) ===\n');
    for s = 1:nSubjects
        fprintf('  %s: %.2f%% (n = %d epochs)\n', ...
                subject_names{s}, fold_acc(s), per_subject_counts(s));
    end
    fprintf('\n  Mean accuracy: %.2f%%\n', mean_acc);
    fprintf('  Std  accuracy: %.2f%%\n', std_acc);

    fprintf('\nConfusion matrix (rows = true, cols = predicted):\n');
    disp(conf_mat);

    % Put everything in a results struct
    results = struct();
    results.fold_acc      = fold_acc;
    results.mean_acc      = mean_acc;
    results.std_acc       = std_acc;
    results.conf_mat      = conf_mat;
    results.subject_names = subject_names;
end
