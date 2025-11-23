function [preprocessed_data, labels] = preprocess(dataDir)
%% PREPROCESS  Load and preprocess EEG data based on current iteration.
%
% Inputs:
%   dataDir  - folder that contains R1..R10 EDF/XML files (DATA_DIR)
%
% Outputs:
%   preprocessed_data - preprocessed EEG epochs [nEpochs x nSamplesPerEpoch]
%   labels            - sleep stage labels [nEpochs x 1]
%
% Note: This function expects config variables in the *base* workspace:
%   CURRENT_ITERATION, LOW_PASS_FILTER_FREQ

    % Get config variables from base workspace (config.m is a script)
    try
        CURRENT_ITERATION    = evalin('base', 'CURRENT_ITERATION');
        LOW_PASS_FILTER_FREQ = evalin('base', 'LOW_PASS_FILTER_FREQ');
    catch
        error(['Config variables not found. Make sure config.m has been ', ...
               'run in the base workspace before calling preprocess.']);
    end

    fprintf('Preprocessing data for iteration %d...\n', CURRENT_ITERATION);

    % 1) Load EEG epochs + labels using your load_training_data
    [eeg_data, labels] = load_training_data(dataDir, []);

    % eeg_data: [nEpochs x nSamplesPerEpoch]
    % labels:   [nEpochs x 1]

    % EEG sampling rate (from your EDFs, EEGsec = 125 Hz)
    fs = 125;

    %% --- Iteration 1: Basic preprocessing ---
    if CURRENT_ITERATION == 1
        X = eeg_data;   % [nEpochs x nSamples]

        % 1. "High-pass": subtract a slow moving average (removes drift)
        X = simple_highpass(X, 0.5, fs);

        % 2. "Notch": we skip proper notch (no Signal Processing Toolbox)
        %    but you could implement something custom later if needed.
        X = simple_notch_placeholder(X, 60, fs);

        % 3. "Low-pass": moving average smoothing
        X = simple_lowpass(X, LOW_PASS_FILTER_FREQ, fs);

        preprocessed_data = X;

    %% --- Iteration 2: Enhanced preprocessing ---
    elseif CURRENT_ITERATION == 2
    %% --- Iteration 2: Enhanced preprocessing ---
    elseif CURRENT_ITERATION == 2
    % X starts as EEG epochs from load_training_data:
    %   X: [nEpochs x nSamplesPerEpoch] (single EEG channel per epoch)
    X = eeg_data;

    %% 1) Channel selection & (future) rereferencing
    % Right now load_training_data only uses one EEG channel (EEGsec).
    % If later you extend it to multi-channel (EEG, EOG, EMG), you would:
    %   - select a subset of channels here
    %   - re-reference across channels.
    %
    % With only one EEG channel, rereferencing has no effect, so we just
    % keep the structure here as a placeholder.

    % Example placeholder if you later stack channels:
    % [X, labels] = select_channels_and_reref(X, labels);

    %% 2) Band-pass filtering: ~0.5–40 Hz (drift + HF noise)
    % Using toolbox-free approximations:
    %   - high-pass by subtracting slow moving average (remove drift)
    %   - low-pass by moving average smoothing (remove high-frequency noise)
    X = simple_highpass(X, 0.5, fs);   % remove slow drift (eye movements / DC)
    X = simple_lowpass(X, 40,  fs);    % remove muscle / very HF noise

    %% 3) Artifact removal: eye blinks & muscle bursts (heuristics)

    % 3a) Amplitude-based rejection
    % Large, brief excursions → potential blinks or bad contacts.
    amp_thresh = 150;  % microvolts, adjust if needed
    big_amps = abs(X) > amp_thresh;
    % Zero out extreme values (or you could mark entire epochs as bad)
    X(big_amps) = 0;

    % 3b) High-frequency artifact detection (muscle)
    % Approximate "muscle band" by high-passing at ~20 Hz, then computing
    % per-epoch RMS energy.
    X_hf = simple_highpass(X, 20, fs);          % keep mostly fast activity
    hf_energy = sqrt(mean(X_hf.^2, 2));         % one HF energy per epoch
    hf_thresh = 5 * median(hf_energy);          % robust threshold
    bad_hf_epochs = hf_energy > hf_thresh;

    % 3c) Variance-based data quality check
    epoch_var  = var(X, 0, 2);                  % one variance per epoch
    var_thresh = 5 * median(epoch_var);
    bad_var_epochs = epoch_var > var_thresh;

    % Combine artifact flags
    bad_epochs = bad_hf_epochs | bad_var_epochs;

    fprintf('Iteration 2: rejecting %d/%d epochs as artifacts.\n', ...
            nnz(bad_epochs), numel(bad_epochs));

    % Remove bad epochs from both data and labels so they stay aligned
    keep = ~bad_epochs;
    X      = X(keep, :);
    labels = labels(keep);

    %% 4) Data quality checks (post-artifact-removal)
    % (Already partly handled by removing bad epochs above.)
    % You could also check for all-zero epochs, NaNs, etc.
    bad_zero = all(X == 0, 2);
    if any(bad_zero)
        fprintf('Iteration 2: removing %d all-zero epochs after cleaning.\n', nnz(bad_zero));
        keep = ~bad_zero;
        X      = X(keep, :);
        labels = labels(keep);
    end

    %% 5) Normalization / standardization
    % Normalize each epoch to zero-mean, unit-variance (per row).
    % Add small epsilon to avoid division by zero.
    eps_val = 1e-6;
    mu  = mean(X, 2);
    sig = std(X, 0, 2) + eps_val;
    X = (X - mu) ./ sig;

    % Final output for iteration 2
    preprocessed_data = X;


    %% --- Iteration 3+ (placeholder) ---
    elseif CURRENT_ITERATION >= 3
        fprintf('TODO: Implement advanced preprocessing for iteration %d\n', CURRENT_ITERATION);
        preprocessed_data = eeg_data;

    else
        error('Invalid iteration: %d', CURRENT_ITERATION);
    end
end

%% ===== Helper functions (toolbox-free, operate on [N x T]) =====

function Y = simple_lowpass(X, cutoff, fs)
    % SIMPLE_LOWPASS: moving average low-pass
    % cutoff ~ approximate (-3dB) frequency
    % We pick window length ~ 1 / cutoff seconds
    win_sec = 1 / cutoff;          % seconds
    N = max(1, round(win_sec * fs));   % samples
    % Smooth along time dimension (2nd dim)
    Y = movmean(X, N, 2);
end

function Y = simple_highpass(X, cutoff, fs)
    % SIMPLE_HIGHPASS: subtract slow moving average
    % Cutoff ~ approximate high-pass frequency
    win_sec = 1 / cutoff;          % seconds
    N = max(1, round(win_sec * fs));
    trend = movmean(X, N, 2);      % slow component
    Y = X - trend;                 % remove low-frequency trend
end

function Y = simple_notch_placeholder(X, f0, fs)
    % SIMPLE_NOTCH_PLACEHOLDER
    % With no Signal Processing Toolbox, we skip a true notch filter.
    % This just passes the data through unchanged.
    %
    % If you REALLY want to try something:
    % you could estimate & subtract a 50/60 Hz sine from each epoch.
    %
    % For now, keep it simple:
    Y = X;
end
