function main()
%% Main script to run the sleep scoring pipeline.

clc; close all;
clearvars -except config;

% Add src directory and subdirectories to path
addpath(genpath('src'));

% Load configuration (defines paths, iteration, params)
run('config.m');

fprintf('--- Sleep Scoring Pipeline - Iteration %d ---\n', CURRENT_ITERATION);

%% Make sure preprocess() can see config in *base* workspace
% (preprocess.m uses evalin('base', ...))
try
    % Only push what's needed by preprocess.m
    if ~evalin('base','exist(''CURRENT_ITERATION'',''var'')')
        assignin('base','CURRENT_ITERATION', CURRENT_ITERATION);
    end
    if ~evalin('base','exist(''LOW_PASS_FILTER_FREQ'',''var'')')
        assignin('base','LOW_PASS_FILTER_FREQ', LOW_PASS_FILTER_FREQ);
    end
catch ME
    warning('Could not push config to base workspace: %s', ME.message);
end

%% 1) Preprocessing (loads EDF/XML internally from TRAINING_DIR)
preprocessed_data = [];
labels = [];
cache_filename_preprocess = sprintf('preprocessed_data_iter%d.mat', CURRENT_ITERATION);

if USE_CACHE
    s = load_cache(cache_filename_preprocess, CACHE_DIR);
    if ~isempty(s)
        if isstruct(s) && isfield(s,'X') && isfield(s,'y')
            preprocessed_data = s.X;
            labels = s.y;
        else
            preprocessed_data = s; % legacy cache
        end
    end
end

if isempty(preprocessed_data) || isempty(labels)
    [preprocessed_data, labels] = preprocess(TRAINING_DIR);
    if USE_CACHE
        save_cache(struct('X',preprocessed_data,'y',labels), cache_filename_preprocess, CACHE_DIR);
    end
end

fprintf('Preprocessing done. Epochs: %d, Samples/epoch: %d\n', size(preprocessed_data,1), size(preprocessed_data,2));

%% 2) Feature Extraction
features = [];
cache_filename_features = sprintf('features_iter%d.mat', CURRENT_ITERATION);

if USE_CACHE
    tmp = load_cache(cache_filename_features, CACHE_DIR);
    if ~isempty(tmp), features = tmp; end
end

if isempty(features)
    features = extract_features(preprocessed_data);  % wrapper in src/extract_features.m
    if USE_CACHE
        save_cache(features, cache_filename_features, CACHE_DIR);
    end
end

fprintf('Features extracted: %d x %d (epochs x features)\n', size(features,1), size(features,2));

%% 3) Feature Selection
selected_features = select_features(features, labels);
fprintf('Selected features: %d → %d\n', size(features,2), size(selected_features,2));

%% 4) Classification / Evaluation
model = train_classifier(selected_features, labels);

% Save the trained model for inference
model_filename = sprintf('model_iter%d.mat', CURRENT_ITERATION);
save_cache(model, model_filename, CACHE_DIR);
fprintf('Model saved to cache: %s\n', fullfile(CACHE_DIR, model_filename));

%% 5) Visualization
visualize_results(model, selected_features, labels);

%% 6) Report (optional)
try
    generate_report(model, selected_features, labels);
catch ME
    warning('generate_report not available (%s). Skipping.', ME.message);
end

fprintf('--- Pipeline Finished ---\n');
end
