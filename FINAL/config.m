% ===========================
% Project Configuration (MATLAB)
% ===========================

%% ---- Iteration control ----
% 1: Time features + kNN
% 2: Add spectral features + SVM
% 3: Multi-signal + Random Forest
% 4: Final optimization
CURRENT_ITERATION = 1;

%% ---- Caching ----
USE_CACHE = true;          % cache preprocessed data & features
CACHE_DIR = 'cache/';

%% ---- Data paths ----
% Adjust this path to your machine
DATA_DIR     = '/Users/melissajova/Documents/Signal Processing';
TRAINING_DIR = fullfile(DATA_DIR, 'Training');   % R1..R10 with XML
HOLDOUT_DIR  = fullfile(DATA_DIR, 'Holdout');    % 10 unlabeled EDFs
SAMPLE_DIR   = HOLDOUT_DIR;

% Validate paths / create cache
if ~exist(DATA_DIR, 'dir')
    error('Data directory not found: %s\nRun from the project root or fix DATA_DIR.', DATA_DIR);
end
if ~exist(CACHE_DIR, 'dir')
    fprintf('Creating cache directory: %s\n', CACHE_DIR);
    mkdir(CACHE_DIR);
end

%% ---- Signal & epoch parameters ----
FS_EEG     = 125;    % Hz (EEG, EMG typical in this dataset)
FS_EOG     = 125;     % Hz
EPOCH_SEC  = 30;     % seconds per epoch (AASM standard)

%% ---- Preprocessing parameters ----
% Baseline + band limits for Iteration 1/2
HP_CUTOFF_HZ          = 0.5;   % high-pass to remove drift (implemented via simple_highpass)
LP_CUTOFF_HZ          = 40;    % low-pass cutoff for sleep EEG
POWERLINE_FREQ_HZ     = 50;    % EU mains
NOTCH_Q               = 30;    % notch sharpness (if you need it inside design)
LOW_PASS_FILTER_FREQ  = LP_CUTOFF_HZ; % kept for backward compat with your code

%% ---- Feature extraction parameters ----
% Iteration 1: time-domain features (16 per EEG channel) – already implemented
% Iteration 2+: add spectral features (Welch/AR) – configure here later
FEATURES_ITER1_NAMES = { ...
  'mean','median','std','var','rms','min','max','range','skew','kurt', ...
  'zero_cross','hjorth_activity','hjorth_mobility','hjorth_complexity', ...
  'energy','power' ...
};  % purely informational; not used to run

%% ---- Classification hyperparameters ----
switch CURRENT_ITERATION
    case 1
        CLASSIFIER_TYPE  = 'knn';
        KNN_N_NEIGHBORS  = 5;

    case 2
        CLASSIFIER_TYPE  = 'svm';
        SVM_KERNEL       = 'rbf';
        SVM_C            = 1.0;     % tune later
        SVM_KERNEL_SCALE = 'auto';  % or numeric value

    case 3
        CLASSIFIER_TYPE  = 'random_forest';
        RF_N_TREES       = 100;     % TreeBagger trees
        RF_MAX_DEPTH     = [];      % [] = unlimited; consider limiting later
        RF_CLASS_WEIGHT_ALPHA = 1.0;   % try 0, 0.5, 1, 1.5, 2

    case 4
        CLASSIFIER_TYPE      = 'random_forest';
        RF_N_TREES           = 200;
        RF_MAX_DEPTH         = [];  % unlimited
        RF_MIN_SAMPLES_SPLIT = 5;   % (will map to TreeBagger params if you implement)
    otherwise
        error('Invalid CURRENT_ITERATION: %d (must be 1–4).', CURRENT_ITERATION);
end

%% ---- Submission ----
SUBMISSION_FILE = 'submission.csv';

% ===========================
% IMPORTANT:
% This file must NOT run the pipeline.
% Do NOT call preprocess(), feature extraction, plotting, or training here.
% main.m will:
%   - load this config
%   - run preprocess()
%   - run extract_features()
%   - train/evaluate/visualize
% ===========================