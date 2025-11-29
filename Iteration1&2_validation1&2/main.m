%% Main script to run the sleep scoring pipeline.

clc;
close all;
clearvars -except config;

% Add src directory and subdirectories to path
addpath(genpath('src'));

% Load configuration
run('config.m'); % This will load config variables into the base workspace

fprintf('--- Sleep Scoring Pipeline - Iteration %d ---\n', CURRENT_ITERATION);

edf_dir = TRAINING_DIR;
xml_dir = TRAINING_DIR;

%% 1. Load Data
if CURRENT_ITERATION == 1
    % Iteration 1: single “combined” EEG signal (C3 + C4)
    [eeg_data, labels] = load_training_data(edf_dir, xml_dir);
    fprintf('Single-channel EEG data loaded: %d epochs, %d samples/epoch\n', ...
            size(eeg_data,1), size(eeg_data,2));

    % 2. Preprocessing (single-channel)
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

    % 3. Feature Extraction (single-channel, 16 features)
    features = [];
    cache_filename_features = sprintf('features_iter%d.mat', CURRENT_ITERATION);
    if USE_CACHE
        features = load_cache(cache_filename_features, CACHE_DIR);
    end

    if isempty(features)
        features = extract_features(preprocessed_data);   % uses your 16-feature code
        if USE_CACHE
            save_cache(features, cache_filename_features, CACHE_DIR);
        end
    end

else
    % Iteration 2+ : multi-channel data (2 EEG, 2 EOG, +EMG)
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

    % 2. Preprocessing (multi-channel)
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

    % 3. Feature Extraction (multi-channel, 16 features per channel)
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
end

%% 4. Feature Selection
selected_features = select_features(features, labels);

%% 5. Classification
model = train_classifier(selected_features, labels);

% Save the trained model for inference
model_filename = sprintf('model_iter%d.mat', CURRENT_ITERATION);
save_cache(model, model_filename, CACHE_DIR);

%% 6. Visualization
visualize_results(model, selected_features, labels);

%% 7. Report Generation
generate_report(model, selected_features, labels);

fprintf('--- Pipeline Finished ---\n');
