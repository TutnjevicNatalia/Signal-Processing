%% main.m
% Main script to run the sleep scoring pipeline.

clc;
close all;
clearvars -except config;

% Add src directory and subdirectories to path
addpath(genpath('src'));

% Load configuration (defines TRAINING_DIR, HOLDOUT_DIR, CACHE_DIR,
% CURRENT_ITERATION, USE_CACHE, SUBMISSION_FILE, etc.)
run('config.m');

fprintf('--- Sleep Scoring Pipeline - Iteration %d ---\n', CURRENT_ITERATION);

edf_dir = TRAINING_DIR;
xml_dir = TRAINING_DIR;

%% 1. Load Data
if CURRENT_ITERATION == 1
    % ===================== ITERATION 1: single-channel =====================
    [eeg_data, labels] = load_training_data(edf_dir, xml_dir);
    fprintf('Single-channel EEG data loaded: %d epochs, %d samples/epoch\n', ...
            size(eeg_data,1), size(eeg_data,2));

    %% 2. Preprocessing (single-channel)
    preprocessed_data = [];
    cache_filename_preprocess = sprintf('preprocessed_data_iter%d.mat', CURRENT_ITERATION);
    if USE_CACHE
        preprocessed_data = load_cache(cache_filename_preprocess, CACHE_DIR);
    end

    if isempty(preprocessed_data)
        [preprocessed_data, labels] = preprocess(eeg_data, labels);
        if USE_CACHE
            save_cache(preprocessed_data, cache_filename_preprocess, CACHE_DIR);
        end
    end

    %% 3. Feature Extraction (single-channel)
    features = [];
    cache_filename_features = sprintf('features_iter%d.mat', CURRENT_ITERATION);
    if USE_CACHE
        features = load_cache(cache_filename_features, CACHE_DIR);
    end

    if isempty(features)
        features = extract_features(preprocessed_data);
        if USE_CACHE
            save_cache(features, cache_filename_features, CACHE_DIR);
        end
    end

    channel_info = struct();      % not used for iter 1
    channel_info.subject_ids = []; % keep API consistent

else
    % ===================== ITERATION 2+ : multi-channel =====================
    [multi_channel_data, labels, channel_info] = load_training_data(edf_dir, xml_dir);

    fprintf('Multi-channel data loaded:\n');
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

    %% 2. Preprocessing (multi-channel)
    preprocessed_data = [];
    cache_filename_preprocess = sprintf('preprocessed_data_iter%d.mat', CURRENT_ITERATION);
    if USE_CACHE
        preprocessed_data = load_cache(cache_filename_preprocess, CACHE_DIR);
    end

    if isempty(preprocessed_data)
        [preprocessed_data, labels] = preprocess_multichannel(multi_channel_data, labels, channel_info);
        if USE_CACHE
            save_cache(preprocessed_data, cache_filename_preprocess, CACHE_DIR);
        end
    end

        %% 3. Feature Extraction (multi-channel)
    features = [];
    cache_filename_features = sprintf('features_iter%d.mat', CURRENT_ITERATION);
    if USE_CACHE
        features = load_cache(cache_filename_features, CACHE_DIR);
    end

    if isempty(features)
        features = extract_features_multichannel(preprocessed_data, channel_info, CURRENT_ITERATION);
        if USE_CACHE
            save_cache(features, cache_filename_features, CACHE_DIR);
        end
    end

    % Build feature names to match extract_features_multichannel layout
    feature_names = build_feature_names_multichannel(preprocessed_data, CURRENT_ITERATION);

    % --- NEW: subject-wise normalisation BEFORE global scaling (Iter 3+) ---
    if CURRENT_ITERATION >= 3 && isfield(channel_info, 'subject_ids')
        fprintf('Applying subject-wise z-scoring of features (Iteration 3+).\n');
        features = subjectwise_zscore_features(features, channel_info.subject_ids);
    end

end

%% 4. Feature Selection + Standardization + Model Training
fprintf('\n[4] Feature selection and model training...\n');

% select_features should return both reduced features and selected indices
% select_features now also receives feature_names (optional)
[selected_features, sel_idx] = select_features(features, labels, feature_names);
fprintf('  Selected %d features out of %d\n', size(selected_features,2), size(features,2));

% Global standardisation (z-score) based on TRAINING ONLY
mu_train  = mean(selected_features, 1);
sig_train = std(selected_features, 0, 1) + 1e-6;

X_train = (selected_features - mu_train) ./ sig_train;

% Train classifier on all training data
model = train_classifier(X_train, labels);

%% 5. Save the trained model + feature selection info
if ~exist(CACHE_DIR, 'dir')
    mkdir(CACHE_DIR);
end

model_filename = sprintf('model_iter%d.mat', CURRENT_ITERATION);
model_path     = fullfile(CACHE_DIR, model_filename);

save(model_path, 'model', 'sel_idx', 'mu_train', 'sig_train', '-v7.3');
fprintf('  Saved trained model and feature selection info to %s\n', model_path);

%% 6. Visualization (optional, uses selected & standardized features)
visualize_results(model, X_train, labels);

%% 7. Report Generation (optional)
generate_report(model, X_train, labels);

fprintf('--- Pipeline Finished ---\n');
