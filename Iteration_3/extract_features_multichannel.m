function features = extract_features_multichannel(preprocessed_data, channel_info, CURRENT_ITERATION)
%% EXTRACT_FEATURES_MULTICHANNEL
% Iter 2:
%   EEG, EOG → full time+spectral features via feature_extraction_extract_features
%
% Iter 3:
%   EEG → same full feature set
%   EOG → simplified REM-oriented features (~6 per channel)
%   EMG → muscle-tone features (~3 per channel)

    fprintf('Extracting multi-channel features for iteration %d...\n', CURRENT_ITERATION);

    feats_all = [];

    %% === EEG: full feature set (time + spectral) ===
    if isfield(preprocessed_data,'eeg')
        X = preprocessed_data.eeg;  % [epochs x nEEG x samples]
        [nEpochs, nEEG, ~] = size(X);
        feats_eeg = [];

        for ch = 1:nEEG
            sig = squeeze(X(:, ch, :));  % [epochs x samples]
            feats_ch = feature_extraction_extract_features(sig, CURRENT_ITERATION);
            feats_eeg = [feats_eeg, feats_ch]; %#ok<AGROW>
        end

        feats_all = [feats_all, feats_eeg]; %#ok<AGROW>
        fprintf('  EEG features: %d channels -> %d features\n', nEEG, size(feats_eeg,2));
    end

    %% === EOG ===
    if isfield(preprocessed_data,'eog')
        X = preprocessed_data.eog;  % [epochs x nEOG x samples]
        [nEpochs, nEOG, ~] = size(X);

        if CURRENT_ITERATION >= 3
            % Iter 3: REM-oriented features
            fs_eog = channel_info.eog.fs;
            feats_eog = [];
            for ch = 1:nEOG
                sig = squeeze(X(:, ch, :));  % [epochs x samples]
                feats_ch = compute_eog_features(sig, fs_eog);  % [nEpochs x 6]
                feats_eog = [feats_eog, feats_ch]; %#ok<AGROW>
            end
            fprintf('  EOG features (iter3): %d channels -> %d features\n', ...
                nEOG, size(feats_eog,2));
        else
            % Iter 2: same as EEG
            feats_eog = [];
            for ch = 1:nEOG
                sig = squeeze(X(:, ch, :));
                feats_ch = feature_extraction_extract_features(sig, CURRENT_ITERATION);
                feats_eog = [feats_eog, feats_ch]; %#ok<AGROW>
            end
            fprintf('  EOG features: %d channels -> %d features\n', nEOG, size(feats_eog,2));
        end

        feats_all = [feats_all, feats_eog]; %#ok<AGROW>
    end

    %% === EMG (only from Iteration 3) ===
    if isfield(preprocessed_data,'emg') && CURRENT_ITERATION >= 3
        X = preprocessed_data.emg;  % [epochs x nEMG x samples]
        [nEpochs, nEMG, ~] = size(X);
        fs_emg = channel_info.emg.fs;
        feats_emg = [];

        for ch = 1:nEMG
            sig = squeeze(X(:, ch, :));  % [epochs x samples]
            feats_ch = compute_emg_features(sig, fs_emg);  % [nEpochs x 3]
            feats_emg = [feats_emg, feats_ch]; %#ok<AGROW>
        end

        feats_all = [feats_all, feats_emg]; %#ok<AGROW>
        fprintf('  EMG features (iter3): %d channels -> %d features\n', ...
            nEMG, size(feats_emg,2));
    end

    features = feats_all;
    fprintf('Total multi-channel features: %d (per epoch)\n', size(features,2));
end

%% ===== EOG features for Iteration 3 =====
% X: [nEpochs x nSamples], fs in Hz
% Returns [nEpochs x 6]:
%   1) peak amplitude         (max |x|)
%   2) variance
%   3) RMS
%   4) zero-crossing rate
%   5) REM peak rate (peaks/s after 0.5 Hz high-pass)
%   6) fraction of samples above threshold in HP signal
function feats = compute_eog_features(X, fs)
    [nEpochs, nSamples] = size(X);
    feats = zeros(nEpochs, 6);

    for i = 1:nEpochs
        epoch = X(i,:);
        epoch = epoch - mean(epoch);

        peak_amp = max(abs(epoch));
        v        = var(epoch, 0, 2);
        rms_val  = sqrt(mean(epoch.^2));

        % zero-crossing rate
        s = sign(epoch);
        s(s == 0) = 1;
        zc = sum(s(1:end-1).*s(2:end) < 0) / nSamples;

        % high-pass ~0.5 Hz via moving-average trend removal
        win_sec = 1 / 0.5;
        N_hp = max(1, round(win_sec * fs));
        trend = movmean(epoch, N_hp, 2);
        epoch_hp = epoch - trend;

        sigma_hp = std(epoch_hp);
        if sigma_hp <= 0
            rem_rate   = 0;
            frac_above = 0;
        else
            thr = 0.5 * sigma_hp;

            % try findpeaks (if Signal Toolbox present), else threshold crossings
            try
                [pks, ~] = findpeaks(abs(epoch_hp), ...
                                     'MinPeakHeight', thr, ...
                                     'MinPeakDistance', round(0.1*fs));
                rem_rate = numel(pks) / (nSamples/fs);  % peaks per second
            catch
                above = abs(epoch_hp) > thr;
                rem_rate = sum(diff(above) == 1) / (nSamples/fs);
            end

            frac_above = mean(abs(epoch_hp) > thr);
        end

        feats(i,:) = [peak_amp, v, rms_val, zc, rem_rate, frac_above];
    end
end

%% ===== EMG features for Iteration 3 =====
% X: [nEpochs x nSamples], fs in Hz
% Returns [nEpochs x 3]:
%   1) total power (mean(x^2))
%   2) variance
%   3) high-freq power proxy (>~20 Hz)
function feats = compute_emg_features(X, fs)
    [nEpochs, nSamples] = size(X);
    feats = zeros(nEpochs, 3);

    for i = 1:nEpochs
        epoch = X(i,:);
        epoch = epoch - mean(epoch);

        total_power = mean(epoch.^2);
        v           = var(epoch, 0, 2);

        % crude high-freq component: subtract ~20 Hz low-pass
        N_lp = max(1, round(fs / 20));   % ~20 Hz
        low  = movmean(epoch, N_lp, 2);
        hf   = epoch - low;
        hf_power = mean(hf.^2);

        feats(i,:) = [total_power, v, hf_power];
    end
end
