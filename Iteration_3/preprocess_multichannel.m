function [preprocessed_data, labels] = preprocess_multichannel(multi_channel_data, labels, channel_info)
%% PREPROCESS_MULTICHANNEL  Preprocess EEG/EOG/EMG for Iteration 2+.
%
% Input:
%   multi_channel_data.eeg : [nEpochs x nEEG x nSamplesEEG]
%   multi_channel_data.eog : [nEpochs x nEOG x nSamplesEOG] (optional)
%   multi_channel_data.emg : [nEpochs x nEMG x nSamplesEMG] (optional)
%   labels                 : [nEpochs x 1]
%
% Output:
%   preprocessed_data : struct with same fields as multi_channel_data
%   labels            : (unchanged for now)

    try
        CURRENT_ITERATION    = evalin('base', 'CURRENT_ITERATION');
        LOW_PASS_FILTER_FREQ = evalin('base', 'LOW_PASS_FILTER_FREQ');
        FS_EEG               = evalin('base', 'FS_EEG');
        FS_EOG               = evalin('base', 'FS_EOG');
    catch
        error(['Config variables not found. Make sure config.m has been ', ...
               'run in the base workspace before calling preprocess_multichannel.']);
    end

    % EMG fs (fallback to EEG if not defined)
    try
        FS_EMG = evalin('base', 'FS_EMG');
    catch
        FS_EMG = FS_EEG;
    end

    fprintf('Preprocessing multi-channel data for iteration %d...\n', CURRENT_ITERATION);

    preprocessed_data = struct();

    %% === EEG ===
    if isfield(multi_channel_data,'eeg')
        X = multi_channel_data.eeg;  % [epochs x nEEG x samples]
        [nEpochs, nEEG, ~] = size(X);
        Xout = zeros(size(X));
        fs = FS_EEG;

        for ch = 1:nEEG
            sig = squeeze(X(:, ch, :));  % [epochs x samples]

            % Basic pipeline as in Iteration 1
            sig = simple_highpass(sig, 0.5, fs);
            sig = simple_notch(sig, 50, fs);
            sig = simple_lowpass(sig, LOW_PASS_FILTER_FREQ, fs);

            Xout(:, ch, :) = sig;
        end
        preprocessed_data.eeg = Xout;
    end

    %% === EOG ===
    if isfield(multi_channel_data,'eog')
        X = multi_channel_data.eog;
        [~, nEOG, ~] = size(X);
        Xout = zeros(size(X));
        fs = FS_EOG;

        for ch = 1:nEOG
            sig = squeeze(X(:, ch, :));  % [epochs x samples]

            % low-freq eye movements, keep up to ~15 Hz
            sig = simple_highpass(sig, 0.1, fs);
            sig = simple_lowpass(sig, 15,  fs);

            Xout(:, ch, :) = sig;
        end
        preprocessed_data.eog = Xout;
    end

    %% === EMG ===
    if isfield(multi_channel_data,'emg')
        X = multi_channel_data.emg;
        [~, nEMG, ~] = size(X);
        Xout = zeros(size(X));
        fs = FS_EMG;

        for ch = 1:nEMG
            sig = squeeze(X(:, ch, :));  % [epochs x samples]

            % EMG: high-frequency band
            sig = simple_highpass(sig, 10, fs);
            sig = simple_lowpass(sig, 40, fs);

            Xout(:, ch, :) = sig;
        end
        preprocessed_data.emg = Xout;
    end

    %% === Iteration 3 artefact handling ===
    if CURRENT_ITERATION >= 3
        % 1) EOG artefact regression: EEG = beta * EOG + residual → keep residual
        if isfield(preprocessed_data,'eeg') && isfield(preprocessed_data,'eog')
            fprintf('  Iter3: removing EOG artefacts from EEG via regression...\n');
            preprocessed_data.eeg = regress_out_eog(preprocessed_data.eeg, preprocessed_data.eog);
        end

        % 2) EMG-based adaptive low-pass on EEG (for high EMG power epochs)
        if isfield(preprocessed_data,'emg') && isfield(preprocessed_data,'eeg')
            fprintf('  Iter3: EMG-based adaptive low-pass on EEG...\n');
            preprocessed_data.eeg = emg_adaptive_lowpass(preprocessed_data.eeg, preprocessed_data.emg, FS_EEG);
        end
    end

    % (You can add artifact rejection & normalization per channel here later.)
end

%% ===== Helper filters (same as before) =====

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
    % Using designfilt (Signal Processing Toolbox installed)
    Q  = 30;
    bw = f0 / Q;
    f1 = f0 - bw/2;
    f2 = f0 + bw/2;

    d = designfilt('bandstopiir', ...
        'FilterOrder', 2, ...
        'HalfPowerFrequency1', f1, ...
        'HalfPowerFrequency2', f2, ...
        'DesignMethod','butter', ...
        'SampleRate', fs);

    Y = zeros(size(X));
    for i = 1:size(X,1)
        Y(i,:) = filtfilt(d, X(i,:));
    end
end

%% ===== Iteration 3: EOG → EEG regression =====
function eeg_clean = regress_out_eog(eeg_data, eog_data)
    [nEpochs, nEEG, nSamples] = size(eeg_data);
    [nEpochs2, nEOG, nSamples2] = size(eog_data);

    if nEpochs2 ~= nEpochs || nSamples2 ~= nSamples
        warning('EOG/EEG size mismatch, skipping EOG artefact removal.');
        eeg_clean = eeg_data;
        return;
    end

    eeg_clean = eeg_data;

    for e = 1:nEpochs
        % predictors from EOG channels: [nSamples x nEOG]
        X = zeros(nSamples, nEOG);
        for chEog = 1:nEOG
            tmp = squeeze(eog_data(e, chEog, :));
            X(:, chEog) = tmp(:);
        end
        % add constant term
        X_aug = [X, ones(nSamples,1)];

        for chEEG = 1:nEEG
            y = squeeze(eeg_data(e, chEEG, :));
            y = y(:);

            % least-squares fit
            beta = X_aug \ y;
            y_hat = X_aug * beta;
            resid = y - y_hat;

            eeg_clean(e, chEEG, :) = resid;
        end
    end
end

%% ===== Iteration 3: EMG-based adaptive low-pass =====
function eeg_out = emg_adaptive_lowpass(eeg_data, emg_data, fs)
    [nEpochs, nEEG, nSamples] = size(eeg_data);
    [nEpochs2, nEMG, nSamples2] = size(emg_data);

    eeg_out = eeg_data;

    if nEpochs2 ~= nEpochs || nSamples2 ~= nSamples || nEMG < 1
        warning('EMG/EEG size mismatch, skipping EMG-based adaptive filter.');
        return;
    end

    % 1) EMG 20–40 Hz "power" per epoch (from first EMG channel)
    emg_power = zeros(nEpochs,1);
    for e = 1:nEpochs
        x = squeeze(emg_data(e,1,:))';
        % crude 20–40 Hz band: high-pass 20, then low-pass 40
        win_hp = 1/20;
        N_hp   = max(1, round(win_hp*fs));
        trend20 = movmean(x, N_hp, 2);
        x_hp    = x - trend20;

        win_lp = 1/40;
        N_lp   = max(1, round(win_lp*fs));
        x_bp   = movmean(x_hp, N_lp, 2);

        emg_power(e) = mean(x_bp.^2);
    end

    % 2) Threshold = median + 2*std (you can tweak this later)
    thr = median(emg_power) + 2*std(emg_power);
    high_idx = find(emg_power > thr);

    if isempty(high_idx)
        % no very noisy EMG epochs → nothing to do
        return;
    end

    % stronger low-pass cutoff for high-EMG epochs
    cutoff_strong = 20;  % Hz
    win_sec = 1 / cutoff_strong;
    N = max(1, round(win_sec * fs));

    for chEEG = 1:nEEG
        for k = 1:numel(high_idx)
            e = high_idx(k);
            y = squeeze(eeg_out(e, chEEG, :))';
            y_f = movmean(y, N, 2);   % strong smoothing
            eeg_out(e, chEEG, :) = y_f;
        end
    end
end
