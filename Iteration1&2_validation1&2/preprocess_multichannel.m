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

    fprintf('Preprocessing multi-channel data for iteration %d...\n', CURRENT_ITERATION);

    preprocessed_data = struct();

    %% EEG
    if isfield(multi_channel_data,'eeg')
        X = multi_channel_data.eeg;  % [epochs x nEEG x samples]
        [nEpochs, nEEG, nSamples] = size(X);
        Xout = zeros(size(X));
        fs = FS_EEG;

        for ch = 1:nEEG
            sig = squeeze(X(:, ch, :));  % [epochs x samples]

            % Basic pipeline as in single-channel Iteration 1
            sig = simple_highpass(sig, 0.5, fs);
            sig = simple_notch(sig, 50,   fs);
            sig = simple_lowpass(sig, LOW_PASS_FILTER_FREQ, fs);

            Xout(:, ch, :) = sig;
        end
        preprocessed_data.eeg = Xout;
    end

    %% EOG
    if isfield(multi_channel_data,'eog')
        X = multi_channel_data.eog;  % [epochs x nEOG x samples]
        [nEpochs, nEOG, nSamples] = size(X);
        Xout = zeros(size(X));
        fs = FS_EOG;

        for ch = 1:nEOG
            sig = squeeze(X(:, ch, :));  % [epochs x samples]

            % EOG: low-frequency eye movements
            sig = simple_highpass(sig, 0.1, fs);
            sig = simple_lowpass(sig, 15,  fs);

            Xout(:, ch, :) = sig;
        end
        preprocessed_data.eog = Xout;
    end

    %% EMG – **only for iteration 3+**
    if CURRENT_ITERATION >= 3 && isfield(multi_channel_data,'emg')
        X = multi_channel_data.emg;  % [epochs x nEMG x samples]
        [nEpochs, nEMG, nSamples] = size(X);
        Xout = zeros(size(X));

        % Assume EMG fs ~ EEG fs (125 Hz) unless you have a separate FS_EMG
        fs_emg = FS_EEG;

        for ch = 1:nEMG
            sig = squeeze(X(:, ch, :));  % [epochs x samples]

            % EMG: higher-frequency band
            sig = simple_highpass(sig, 10, fs_emg);
            sig = simple_lowpass(sig, 40, fs_emg);

            Xout(:, ch, :) = sig;
        end
        preprocessed_data.emg = Xout;
    end

    % (You can add artifact rejection & normalization per channel here later.)
end

%% Helper filters (same as in single-channel preprocess)

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
