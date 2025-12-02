function results = validate_iteration3_loso()
% VALIDATE_ITERATION3_LOSO
%   LOSO cross-validation for Iteration 3 using:
%     - EEG + EOG + EMG (multi-signal, with artefact removal in preprocess_multichannel)
%     - time + spectral EEG features + EOG/EMG features
%     - feature selection (target 40 features via select_features.m)
%     - Random Forest classifier (see train_classifier.m)
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
%           .conf_mat      [5x5]   (rows=true, cols=pred)
%           .subject_names {1x10}

    %% Ensure config is loaded and iteration 3 is set
    run('config.m');
    CURRENT_ITERATION = 3;
    assignin('base','CURRENT_ITERATION', CURRENT_ITERATION);

    % Set classifier to Random Forest for this validation
    CLASSIFIER_TYPE = 'random_forest';
    RF_N_TREES      = 200;   % as in main
    RF_MAX_DEPTH    = [];    % [] = no max depth, or set e.g. 20
    RF_MIN_LEAF     = 5;     % to prevent overfitting

    assignin('base','CLASSIFIER_TYPE', CLASSIFIER_TYPE);
    assignin('base','RF_N_TREES',      RF_N_TREES);
    assignin('base','RF_MAX_DEPTH',    RF_MAX_DEPTH);
    assignin('base','RF_MIN_LEAF',     RF_MIN_LEAF);

    fprintf('=== Iteration 3 LOSO: preprocessing ALL data once ===\n');

    %% 1) Load full multi-channel data
    edf_dir = TRAINING_DIR;
    xml_dir = TRAINING_DIR;

    [mc_data, labels, channel_info] = load_training_data(edf_dir, xml_dir);

    fprintf('Preprocessing multi-channel data for iteration 3...\n');
    pp_data = preprocess_multichannel(mc_data, labels, channel_info);

    fprintf('Extracting multi-channel features for ALL epochs (iteration 3)...\n');
    features_all = extract_features_multichannel(pp_data, channel_info, CURRENT_ITERATION);
    labels_all   = labels(:);

    % Debug sanity check
    fprintf('\n=== Debug check before LOSO (Iter 3) ===\n');
    fprintf('  Features size: %d epochs x %d features\n', size(features_all,1), size(features_all,2));
    fprintf('  NaNs in features? %d\n', any(isnan(features_all),'all'));
    fprintf('  Infs in features? %d\n', any(isinf(features_all),'all'));
    fprintf('  Min feature value: %.3f, Max: %.3f\n', ...
        min(features_all,[],'all'), max(features_all,[],'all'));
    fprintf('  Unique labels: %s\n', sprintf('%d ', unique(labels_all)));
    disp('  Label distribution:');
    tabulate(labels_all);

    % Subject IDs from channel_info (created by load_training_data)
    subject_ids   = channel_info.subject_ids(:);    % numeric: 1..10
    subject_names = channel_info.subject_names;     % {'R1',...,'R10'}
    nSubjects     = numel(subject_names);

    %% 2) Storage for fold metrics
    fold_acc       = zeros(nSubjects,1);
    fold_kappa     = zeros(nSubjects,1);
    fold_f1_macro  = zeros(nSubjects,1);
    conf_mat_total = zeros(5,5);  % 5 classes: 0..4

    fprintf('\n=== Iteration 3 LOSO Cross-Validation (EEG+EOG+EMG, RF + feature selection) ===\n\n');

    %% 3) LOSO loop
    for s = 1:nSubjects
        test_idx  = (subject_ids == s);
        train_idx = ~test_idx;

        X_train_full = features_all(train_idx, :);
        X_test_full  = features_all(test_idx,  :);
        y_train      = labels_all(train_idx);
        y_test       = labels_all(test_idx);

        % Standardize using training fold only
        mu  = mean(X_train_full, 1);
        sig = std(X_train_full, 0, 1) + 1e-6;
        X_train_sc = (X_train_full - mu) ./ sig;
        X_test_sc  = (X_test_full  - mu) ./ sig;

        % --- Feature selection on TRAIN ONLY ---
        [X_train_sel, sel_idx] = select_features(X_train_sc, y_train);
        X_test_sel = X_test_sc(:, sel_idx);

        fprintf('--- Fold %d / %d: leaving out subject %s ---\n', ...
            s, nSubjects, subject_names{s});
        fprintf('  Features after selection: %d\n', size(X_train_sel,2));

        % --- Train Random Forest via train_classifier ---
        % (CLASSIFIER_TYPE and RF_* already set in base)
        model = train_classifier(X_train_sel, y_train);

        % --- Predict on held-out subject ---
        y_pred = predict(model, X_test_sel);
        y_pred = y_pred(:);
        y_test = y_test(:);

        % Metrics
        acc = mean(y_pred == y_test);
        kap = cohen_kappa_local(y_test, y_pred);
        f1m = macro_f1_local(y_test, y_pred, 0:4);

        fold_acc(s)      = acc;
        fold_kappa(s)    = kap;
        fold_f1_macro(s) = f1m;

        % Confusion matrix for this subject
        cm = confusionmat(y_test, y_pred, 'Order', 0:4);
        conf_mat_total = conf_mat_total + cm;

        fprintf('  Subject %s: acc=%.2f%% (n = %d epochs), kappa = %.3f, F1-macro = %.3f\n\n', ...
            subject_names{s}, 100*acc, numel(y_test), kap, f1m);
    end

    %% 4) Aggregate and print summary
    mean_acc      = mean(fold_acc);
    std_acc       = std(fold_acc);
    mean_kappa    = mean(fold_kappa);
    std_kappa     = std(fold_kappa);
    mean_f1_macro = mean(fold_f1_macro);
    std_f1_macro  = std(fold_f1_macro);

    fprintf('=== LOSO Results (Iteration 3, EEG+EOG+EMG+RF+FeatureSelection) ===\n');
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

    %% 5) Pack results in struct
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

%% ===== Local helper: Cohen's kappa =====
function k = cohen_kappa_local(y_true, y_pred)
    y_true = y_true(:);
    y_pred = y_pred(:);
    classes = unique([y_true; y_pred]);
    cm = confusionmat(y_true, y_pred, 'Order', classes);
    n = sum(cm(:));
    po = trace(cm) / n;
    pe = sum(sum(cm,1) .* sum(cm,2)) / (n^2);
    k = (po - pe) / (1 - pe + eps);
end

%% ===== Local helper: Macro F1 =====
function f1_macro = macro_f1_local(y_true, y_pred, class_list)
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
