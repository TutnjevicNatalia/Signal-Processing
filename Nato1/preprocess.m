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

    % Save original for plotting
    X_raw = eeg_data;

    % eeg_data: [nEpochs x nSamplesPerEpoch]
    % labels:   [nEpochs x 1]

    % EEG sampling rate
    fs = 125;

    %% --- Iteration 1: Basic preprocessing ---
    if CURRENT_ITERATION == 1
        X = eeg_data;   % [nEpochs x nSamples]

        % 1. High-pass approximation
        X = simple_highpass(X, 0.5, fs);

        % 2. Real 50 Hz notch filter
        X = simple_notch(X, 50, fs);

        % 3. Low-pass approximation
        X = simple_lowpass(X, LOW_PASS_FILTER_FREQ, fs);

        preprocessed_data = X;

    %% --- Iteration 2: Enhanced preprocessing ---
    elseif CURRENT_ITERATION == 2
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
        
        % From iteration 1
        X = simple_highpass(X, 0.5, fs);
        X = simple_notch(X, 50, fs);
        X = simple_lowpass(X, LOW_PASS_FILTER_FREQ, fs);


        % 2) Band-pass approximations
        X = simple_highpass(X, 0.5, fs);
        X = simple_lowpass(X, 40,  fs);

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

        % 5) Normalize each epoch
        eps_val = 1e-6;
        mu  = mean(X, 2);
        sig = std(X, 0, 2) + eps_val;
        X = (X - mu) ./ sig;

        preprocessed_data = X;

    %% --- Iteration 3+ (placeholder) ---
    elseif CURRENT_ITERATION >= 3
        fprintf('TODO: Implement advanced preprocessing for iteration %d\n', CURRENT_ITERATION);
        preprocessed_data = eeg_data;

    else
        error('Invalid iteration: %d', CURRENT_ITERATION);
    end

    %% ============================
    %   PLOT BEFORE & AFTER
    % ============================
    % Pick first epoch for visualization
    raw_signal   = X_raw(1, :);            % original
    clean_signal = preprocessed_data(1, :); % preprocessed

    t = (0:length(raw_signal)-1) / fs;    % time axis in seconds

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

end  % end of preprocess function

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

%% ===== REAL 50 HZ NOTCH FILTER (designfilt version) =====
function Y = simple_notch(X, f0, fs)
    Q = 30;                 % notch sharpness
    bw = f0 / Q;            % bandwidth around f0
    f1 = f0 - bw/2;         % lower -3 dB point
    f2 = f0 + bw/2;         % upper -3 dB point

    d = designfilt('bandstopiir', ...
        'FilterOrder', 2, ...
        'HalfPowerFrequency1', f1, ...
        'HalfPowerFrequency2', f2, ...
        'DesignMethod','butter', ...
        'SampleRate', fs);

    % Zero-phase filter each epoch
    Y = zeros(size(X));
    for i = 1:size(X,1)
        Y(i,:) = filtfilt(d, X(i,:));
    end
end