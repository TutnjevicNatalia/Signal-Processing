function features = extract_features_multichannel(preprocessed_data, channel_info, CURRENT_ITERATION)
%% EXTRACT_FEATURES_MULTICHANNEL
% Multi-channel feature extraction for Iteration 2 and 3.
%
% Iteration 2:
%   - EEG: feature_extraction_extract_features  (29 features per channel)
%   - EOG: feature_extraction_extract_features  (29 features per channel)
%   - EMG: NOT used
%   => 2 EEG + 2 EOG = 4 * 29 = 116 features
%
% Iteration 3:
%   - EEG:  29 base + 20 wavelet = 49 per channel -> 2 * 49 = 98
%   - EOG:   29 base + 10 wavelet = 16 per ch -> 2 * 16 = 32
%   - EMG:   3 base (compute_emg_features) + 10 wavelet = 13 per ch -> 1 * 13 = 13
%   => 98 + 32 + 13 = 189 total features per epoch

    if nargin < 3 || isempty(CURRENT_ITERATION)
        try
            CURRENT_ITERATION = evalin('base','CURRENT_ITERATION');
        catch
            CURRENT_ITERATION = 2;
        end
    end

    fprintf('Extracting multi-channel features for iteration %d.\n', CURRENT_ITERATION);
    feats_all = [];

    %% === EEG ===
    if isfield(preprocessed_data,'eeg') && ~isempty(preprocessed_data.eeg)
        X = preprocessed_data.eeg;            % [epochs x nEEG x samples]
        [nEpochs, nEEG, ~] = size(X);

        feats_eeg = [];

        for ch = 1:nEEG
            sig = squeeze(X(:, ch, :));       % [epochs x samples]

            % Base time + spectral features (per your original design: 29)
            feats_base = feature_extraction_extract_features(sig, CURRENT_ITERATION);  % [nEpochs x 29]

            if CURRENT_ITERATION >= 3
                % Extra 20 wavelet features per EEG channel
                feats_wav = compute_wavelet_features_basic(sig);  % [nEpochs x 20]
                feats_ch  = [feats_base, feats_wav];              % [nEpochs x 49]
            else
                feats_ch  = feats_base;                           % [nEpochs x 29]
            end

            feats_eeg = [feats_eeg, feats_ch]; %#ok<AGROW>
        end

        feats_all = [feats_all, feats_eeg]; %#ok<AGROW>
        fprintf('  EEG features: %d channels -> %d features\n', nEEG, size(feats_eeg,2));
    end

    %     %% === EOG === 189 features 
if isfield(preprocessed_data,'eog') && ~isempty(preprocessed_data.eog)
    X = preprocessed_data.eog;            % [epochs x nEOG x samples]
    [nEpochs, nEOG, ~] = size(X);

    if CURRENT_ITERATION >= 3
        % Iteration 3: keep 29 "base" features (same as Iteration 2) + 10 wavelet
        feats_eog = [];

        for ch = 1:nEOG
            sig = squeeze(X(:, ch, :));         % [epochs x samples]

            % Base time + spectral features (same extractor as EEG/EOG Iteration 2)
            feats_base = feature_extraction_extract_features(sig, CURRENT_ITERATION);  % [nEpochs x 29]

            % Extra: 10 wavelet features per EOG channel
            feats_wav_all = compute_wavelet_features_basic(sig); % [nEpochs x 20]
            feats_wav     = feats_wav_all(:, 1:10);              % keep first 10

            % Total per EOG channel in Iteration 3: 29 + 10 = 39
            feats_ch  = [feats_base, feats_wav];                 % [nEpochs x 39]
            feats_eog = [feats_eog, feats_ch]; %#ok<AGROW>
        end

        fprintf('  EOG features (iter3): %d channels -> %d features\n', ...
                nEOG, size(feats_eog,2));
    else
        % Iteration 2: EOG treated like EEG (29 per channel)
        feats_eog = [];
        for ch = 1:nEOG
            sig = squeeze(X(:, ch, :));    % [epochs x samples]
            feats_ch = feature_extraction_extract_features(sig, CURRENT_ITERATION); % [nEpochs x 29]
            feats_eog = [feats_eog, feats_ch]; %#ok<AGROW>
        end
        fprintf('  EOG features (iter2): %d channels -> %d features\n', ...
                nEOG, size(feats_eog,2));
    end

    feats_all = [feats_all, feats_eog]; %#ok<AGROW>
end


    %% === EMG (only from Iteration 3) ===
    if isfield(preprocessed_data,'emg') && ~isempty(preprocessed_data.emg) && CURRENT_ITERATION >= 3
        X = preprocessed_data.emg;            % [epochs x nEMG x samples]
        [nEpochs, nEMG, ~] = size(X);
        fs_emg = channel_info.emg.fs;

        feats_emg = [];

        for ch = 1:nEMG
            sig = squeeze(X(:, ch, :));       % [epochs x samples]

            % Base EMG features (3) – your original function
            feats_base = compute_emg_features(sig, fs_emg);         % [nEpochs x 3]

            % Extra: 10 wavelet features per EMG channel
            feats_wav_all = compute_wavelet_features_basic(sig);    % [nEpochs x 20]
            feats_wav     = feats_wav_all(:, 1:10);                 % keep first 10

            feats_ch = [feats_base, feats_wav];                     % [nEpochs x 13]
            feats_emg = [feats_emg, feats_ch]; %#ok<AGROW>
        end

        feats_all = [feats_all, feats_emg]; %#ok<AGROW>
        fprintf('  EMG features (iter3): %d channels -> %d features\n', ...
                nEMG, size(feats_emg,2));
    end

    features = feats_all;
    fprintf('Total multi-channel features: %d (per epoch)\n', size(features,2));
end


%% ===== Wavelet features (20 per channel) =====
% X: [nEpochs x nSamples]
% Returns [nEpochs x 20]:
%   For each of 5 subbands (A4, D4, D3, D2, D1):
%      1) energy (mean coeff^2)
%      2) RMS
%      3) log-energy
%      4) Shannon entropy
function feats = compute_wavelet_features_basic(X)
    [nEpochs, ~] = size(X);
    nBands = 5;                    % A4, D4, D3, D2, D1
    feats  = zeros(nEpochs, 4*nBands);  % 4 stats * 5 bands = 20 features

    for i = 1:nEpochs
        epoch = double(X(i,:));
        epoch = epoch - mean(epoch);    % remove DC

        try
            level = 4;
            wname = 'db4';

            % Wavelet decomposition
            [c, l] = wavedec(epoch, level, wname);
            % Approximation and details
            ca4 = appcoef(c, l, wname, level);
            cd4 = detcoef(c, l, 4);
            cd3 = detcoef(c, l, 3);
            cd2 = detcoef(c, l, 2);
            cd1 = detcoef(c, l, 1);
            bands = {ca4, cd4, cd3, cd2, cd1};
        catch
            % If Wavelet Toolbox not available, return zeros for this epoch
            continue;
        end

        E   = zeros(1, nBands);
        RMS = zeros(1, nBands);
        LE  = zeros(1, nBands);
        ENT = zeros(1, nBands);

        for b = 1:nBands
            % ✅ MATLAB cell indexing with {}
            coeffs = bands{b};
            e = mean(coeffs.^2);
            E(b)   = e;
            RMS(b) = sqrt(e);
            LE(b)  = log10(e + eps);

            % Shannon entropy of coefficient energy distribution
            p = coeffs.^2;
            p = p / (sum(p) + eps);
            ENT(b) = -sum(p .* log2(p + eps));
        end

        feats(i,:) = [E, RMS, LE, ENT];
    end
end


%% ===== EOG features (base 6 per channel) =====
% X: [nEpochs x nSamples], fs in Hz
% Returns [nEpochs x 6]
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
        N_hp    = max(1, round(win_sec * fs));
        trend   = movmean(epoch, N_hp, 2);
        epoch_hp = epoch - trend;
        sigma_hp = std(epoch_hp);

        if sigma_hp <= 0
            rem_rate   = 0;
            frac_above = 0;
        else
            thr = 0.5 * sigma_hp;

            % try findpeaks, else simple threshold crossings
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


%% ===== EMG features (base 3 per channel) =====
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
