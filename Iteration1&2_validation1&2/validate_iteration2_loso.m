function results = validate_iteration2_loso()
% VALIDATE_ITERATION2_LOSO
%   LOSO cross-validation for Iteration 2 using:
%     - EEG + EOG
%     - time + spectral features
%     - feature selection (variance + correlation + top-40)
%     - SVM (ECOC, RBF)
%
% Output struct:
%   results.fold_acc       [10x1]
%           .fold_kappa    [10x1]
%           .fold_f1_macro [10x1]
%           .mean_acc
%           .std_acc
%           .mean_kappa
%           .std_kappa
%           .mean_f1_macro
%           .std_f1_macro
%           .conf_mat      [5x5]
%           .subject_names {1x10}

    %% Ensure config is loaded and iteration is 2
    run('config.m');
    CURRENT_ITERATION = 2;      % force iteration 2 behavior
    assignin('base', 'CURRENT_ITERATION', CURRENT_ITERATION);

    fprintf('=== Iteration 2 LOSO: preprocessing ALL data once ===\n');

    %% 1) Load full multi-channel data (all 10 subjects)
    edf_dir = TRAINING_DIR;
    xml_dir = TRAINING_DIR;

    [mc_data, labels, channel_info] = load_training_data(edf_dir, xml_dir);

    % These MUST be filled in load_training_data:
    %   channel_info.subject_ids   -> numeric 1..10 per epoch
    %   channel_info.subject_names -> {'R1','R2',...,'R10'}
    subject_ids   = channel_info.subject_ids(:);
    subject_names = channel_info.subject_names;
    nSubjects     = numel(subject_names);

    %% 2) Preprocess all data ONCE
    fprintf('Preprocessing multi-channel data for iteration 2...\n');
    pp_data = preprocess_multichannel(mc_data, labels, channel_info);

    %% 3) Extract features for ALL epochs ONCE
    fprintf('Extracting multi-channel features for ALL epochs (iteration 2)...\n');
    X_all_raw = extract_features_multichannel(pp_data, channel_info, CURRENT_ITERATION);
    y_all     = labels(:);

    [nEpochs, nFeat_raw] = size(X_all_raw);
    fprintf('  Raw feature matrix: %d epochs x %d features\n', nEpochs, nFeat_raw);

    %% 4) Storage for LOSO metrics
    fold_acc       = zeros(nSubjects,1);
    fold_kappa     = zeros(nSubjects,1);
    fold_f1_macro  = zeros(nSubjects,1);
    conf_mat_total = zeros(5,5);   % 5 classes: 0..4

    fprintf('\n=== Iteration 2 LOSO Cross-Validation (EEG+EOG, SVM + feature selection) ===\n\n');

    for s = 1:nSubjects
        test_name = subject_names{s};

        % Boolean masks for this fold
        test_idx  = (subject_ids == s);
        train_idx = ~test_idx;

        X_train_full = X_all_raw(train_idx, :);
        X_test_full  = X_all_raw(test_idx,  :);
        y_train      = y_all(train_idx);
        y_test       = y_all(test_idx);

        %% 4a) Feature selection on TRAIN ONLY
        % CURRENT_ITERATION=2 is already set in base, so select_features
        % will apply variance filter + correlation filter + top-40 ranking.
        [X_train_fs, sel_idx] = select_features(X_train_full, y_train);
        X_test_fs = X_test_full(:, sel_idx);

        fprintf('--- Fold %d / %d: leaving out subject %s ---\n', ...
            s, nSubjects, test_name);
        fprintf('  Features after selection: %d\n', size(X_train_fs,2));

        %% 4b) Scale using TRAIN ONLY (after feature selection)
        mu  = mean(X_train_fs, 1);
        sig = std(X_train_fs, 0, 1) + 1e-6;
        X_train_sc = (X_train_fs - mu) ./ sig;
        X_test_sc  = (X_test_fs  - mu) ./ sig;

        %% 4c) Set classifier config explicitly
        CLASSIFIER_TYPE  = 'svm';
        SVM_KERNEL       = 'rbf';
        SVM_C            = 1.0;
        SVM_KERNEL_SCALE = 'auto';

        assignin('base','CURRENT_ITERATION', CURRENT_ITERATION);
        assignin('base','CLASSIFIER_TYPE', CLASSIFIER_TYPE);
        assignin('base','SVM_KERNEL', SVM_KERNEL);
        assignin('base','SVM_C', SVM_C);
        assignin('base','SVM_KERNEL_SCALE', SVM_KERNEL_SCALE);

        %% 4d) Train SVM on 9 subjects
        model = train_classifier(X_train_sc, y_train);

        %% 4e) Predict on held-out subject
        y_pred = predict(model, X_test_sc);
        y_pred = y_pred(:);
        y_test = y_test(:);

        %% 4f) Metrics
        acc = mean(y_pred == y_test);
        kap = cohen_kappa(y_test, y_pred);
        f1m = macro_f1(y_test, y_pred, 0:4);

        fold_acc(s)      = acc;
        fold_kappa(s)    = kap;
        fold_f1_macro(s) = f1m;

        fprintf('  Subject %s accuracy: %.2f%% (n = %d epochs), kappa = %.3f, F1-macro = %.3f\n\n', ...
            test_name, 100*acc, numel(y_test), kap, f1m);

        %% 4g) Confusion matrix (aggregate)
        cm = confusionmat(y_test, y_pred, 'Order', 0:4);
        conf_mat_total = conf_mat_total + cm;
    end

    %% 5) Aggregate stats
    mean_acc      = mean(fold_acc);
    std_acc       = std(fold_acc);
    mean_kappa    = mean(fold_kappa);
    std_kappa     = std(fold_kappa);
    mean_f1_macro = mean(fold_f1_macro);
    std_f1_macro  = std(fold_f1_macro);

    fprintf('=== LOSO Results (Iteration 2, EEG+EOG+Spectral+SVM+FeatureSelection) ===\n');
    for s = 1:nSubjects
        fprintf('  %s: acc=%.2f%%, kappa=%.3f, F1-macro=%.3f (n = %d epochs)\n', ...
            subject_names{s}, 100*fold_acc(s), fold_kappa(s), fold_f1_macro(s), ...
            sum(subject_ids == s));
    end

    fprintf('\n  Mean accuracy: %.2f%% (± %.2f%%)\n', 100*mean_acc, 100*std_acc);
    fprintf('  Mean kappa:   %.3f (± %.3f)\n', mean_kappa, std_kappa);
    fprintf('  Mean F1-macro: %.3f (± %.3f)\n', mean_f1_macro, std_f1_macro);

    fprintf('\nConfusion matrix (rows = true, cols = predicted):\n');
    disp(conf_mat_total);

    %% 6) Pack results
    results = struct();
    results.fold_acc      = fold_acc;
    results.fold_kappa    = fold_kappa;
    results.fold_f1_macro = fold_f1_macro;
    results.mean_acc      = mean_acc;
    results.std_acc       = std_acc;
    results.mean_kappa    = mean_kappa;
    results.std_kappa     = std_kappa;
    results.mean_f1_macro = mean_f1_macro;
    results.std_f1_macro  = std_f1_macro;
    results.conf_mat      = conf_mat_total;
    results.subject_names = subject_names;
end

%% ===== Helper: Cohen's kappa =====
function k = cohen_kappa(y_true, y_pred)
    y_true = y_true(:);
    y_pred = y_pred(:);
    classes = unique([y_true; y_pred]);
    cm = confusionmat(y_true, y_pred, 'Order', classes);
    n = sum(cm(:));
    po = trace(cm) / n;
    pe = sum(sum(cm,1) .* sum(cm,2)) / (n^2);
    k = (po - pe) / (1 - pe + eps);
end

%% ===== Helper: Macro F1 =====
function f1_macro = macro_f1(y_true, y_pred, class_list)
    y_true = y_true(:);
    y_pred = y_pred(:);
    nC = numel(class_list);
    f1s = nan(1,nC);

    for i = 1:nC
        c = class_list(i);
        tp = sum((y_true == c) & (y_pred == c));
        fp = sum((y_true ~= c) & (y_pred == c));
        fn = sum((y_true == c) & (y_pred ~= c));
        if tp == 0 && fp == 0 && fn == 0
            f1s(i) = NaN;
        else
            prec = tp / (tp + fp + eps);
            rec  = tp / (tp + fn + eps);
            f1s(i) = 2 * prec * rec / (prec + rec + eps);
        end
    end

    f1_macro = nanmean(f1s);
end
