function [preprocessed_data, labels] = preprocess(eeg_data, labels)
%% PREPROCESS  Preprocess EEG epochs based on current iteration.
%
% Inputs:
%   eeg_data  - [nEpochs x nSamplesPerEpoch] raw EEG
%   labels    - [nEpochs x 1] sleep stage labels
%
% Outputs:
%   preprocessed_data - [nEpochs_kept x nSamplesPerEpoch] preprocessed EEG
%   labels            - [nEpochs_kept x 1] updated labels (after artifact removal)
%
% Note: This function expects config variables in the *base* workspace:
%   CURRENT_ITERATION, LOW_PASS_FILTER_FREQ, FS_EEG

    % Get config variables from base workspace (config.m is a script)
    try
        CURRENT_ITERATION    = evalin('base', 'CURRENT_ITERATION');
        LOW_PASS_FILTER_FREQ = evalin('base', 'LOW_PASS_FILTER_FREQ');
        FS_EEG               = evalin('base', 'FS_EEG');
    catch
        error(['Config variables not found. Make sure config.m has been ', ...
               'run in the base workspace before calling preprocess.']);
    end

    fprintf('Preprocessing data for iteration %d...\n', CURRENT_ITERATION);

    % Save original for plotting
    X_raw = eeg_data;
    X     = eeg_data;
    fs    = FS_EEG;   % 125 Hz in your config

    %% --- Iteration 1: Basic preprocessing ---
    if CURRENT_ITERATION == 1
        % 1. High-pass approximation
        X = simple_highpass(X, 0.5, fs);

        % 2. Real 50 Hz notch filter
        X = simple_notch(X, 50, fs);

        % 3. Low-pass approximation
        X = simple_lowpass(X, LOW_PASS_FILTER_FREQ, fs);

        preprocessed_data = X;

    %% --- Iteration 2: Enhanced preprocessing ---
    elseif CURRENT_ITERATION == 2
        % From iteration 1
        X = simple_highpass(X, 0.5, fs);
        X = simple_notch(X, 50, fs);
        X = simple_lowpass(X, LOW_PASS_FILTER_FREQ, fs);

        % Extra band-pass (0.5–40 Hz)
        X = simple_highpass(X, 0.5, fs);
        X = simple_lowpass(X, 40,  fs);

        %% Artifact removal

        % 3a) Amplitude-based rejection
        amp_thresh = 150;  % microvolts, adjust if needed
        big_amps = abs(X) > amp_thresh;
        X(big_amps) = 0;

        % 3b) High-frequency artifact detection (muscle)
        X_hf = simple_highpass(X, 20, fs);
        hf_energy = sqrt(mean(X_hf.^2, 2));
        hf_thresh = 5 * median(hf_energy);
        bad_hf_epochs = hf_energy > hf_thresh;

        % 3c) Variance-based data quality check
        epoch_var  = var(X, 0, 2);
        var_thresh = 5 * median(epoch_var);
        bad_var_epochs = epoch_var > var_thresh;

        % Combine artifact flags
        bad_epochs = bad_hf_epochs | bad_var_epochs;

        fprintf('Iteration 2: rejecting %d/%d epochs as artifacts.\n', ...
                nnz(bad_epochs), numel(bad_epochs));

        keep = ~bad_epochs;
        X      = X(keep, :);
        labels = labels(keep);

        % Remove all-zero epochs
        bad_zero = all(X == 0, 2);
        if any(bad_zero)
            fprintf('Iteration 2: removing %d all-zero epochs after cleaning.\n', nnz(bad_zero));
            keep = ~bad_zero;
            X      = X(keep, :);
            labels = labels(keep);
        end

        % Normalize each epoch
        eps_val = 1e-6;
        mu  = mean(X, 2);
        sig = std(X, 0, 2) + eps_val;
        X = (X - mu) ./ sig;

        preprocessed_data = X;

    %% --- Iteration 3+ (placeholder) ---
    elseif CURRENT_ITERATION >= 3
        fprintf('TODO: Implement advanced preprocessing for iteration %d\n', CURRENT_ITERATION);
        preprocessed_data = X;  % currently just pass-through

    else
        error('Invalid iteration: %d', CURRENT_ITERATION);
    end

    %% Plot first epoch before & after
    raw_signal   = X_raw(1, :);
    clean_signal = preprocessed_data(1, :);

    t = (0:length(raw_signal)-1) / fs;

    figure;
    subplot(2,1,1);
    plot(t, raw_signal);
    title('Raw Signal (Before Preprocessing)');
    xlabel('Time (s)');
    ylabel('Amplitude');

    subplot(2,1,2);
    plot(t, clean_signal);
    title('Clean Signal (After Preprocessing)');
    xlabel('Time (s)');
    ylabel('Amplitude');
end

%% ===== Helper functions (operate on [N x T]) =====

function Y = simple_lowpass(X, cutoff, fs)
    win_sec = 1 / cutoff;
    N = max(1, round(win_sec * fs));
    Y = movmean(X, N, 2);
end

function Y = simple_highpass(X, cutoff, fs)
    win_sec = 1 / cutoff;
    N = max(1, round(win_sec * fs));
    trend = movmean(X, N, 2);
    Y = X - trend;
end

function Y = simple_notch(X, f0, fs)
    Q = 30;
    bw = f0 / Q;
    f1 = f0 - bw/2;
    f2 = f0 + bw/2;

    d = designfilt('bandstopiir', ...
        'FilterOrder', 2, ...
        'HalfPowerFrequency1', f1, ...
        'HalfPowerFrequency2', f2, ...
        'DesignMethod', 'butter', ...
        'SampleRate', fs);

    Y = zeros(size(X));
    for i = 1:size(X,1)
        Y(i,:) = filtfilt(d, X(i,:));
    end
end