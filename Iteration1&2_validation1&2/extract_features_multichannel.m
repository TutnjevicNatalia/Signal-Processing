function features = extract_features_multichannel(preprocessed_data, channel_info, CURRENT_ITERATION)
%% EXTRACT_FEATURES_MULTICHANNEL
% Compute 16 time-domain features per channel and concatenate.
%
% Example:
%   EEG (2 ch) -> 32 features
%   EOG (2 ch) -> 32 features
%   EMG (1 ch) -> 16 features  (only for iteration >= 3)
%   Total      -> 64 or 80 features per epoch depending on iteration.

    % ---- HARD RULE: Iteration 2 must NOT use EMG ----
    if CURRENT_ITERATION == 2 && isfield(preprocessed_data, 'emg')
        preprocessed_data = rmfield(preprocessed_data, 'emg');
    end

    fprintf('Extracting multi-channel features for iteration %d...\n', CURRENT_ITERATION);

    feats_all = [];

    %% EEG
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

    %% EOG
    if isfield(preprocessed_data,'eog')
        X = preprocessed_data.eog;  % [epochs x nEOG x samples]
        [nEpochs, nEOG, ~] = size(X);
        feats_eog = [];
        for ch = 1:nEOG
            sig = squeeze(X(:, ch, :));  % [epochs x samples]
            feats_ch = feature_extraction_extract_features(sig, CURRENT_ITERATION);
            feats_eog = [feats_eog, feats_ch]; %#ok<AGROW>
        end
        feats_all = [feats_all, feats_eog]; %#ok<AGROW>
        fprintf('  EOG features: %d channels -> %d features\n', nEOG, size(feats_eog,2));
    end

    %% EMG – **only from iteration 3 onwards**
    if CURRENT_ITERATION >= 3 && isfield(preprocessed_data,'emg')
        X = preprocessed_data.emg;  % [epochs x nEMG x samples]
        [nEpochs, nEMG, ~] = size(X);
        feats_emg = [];
        for ch = 1:nEMG
            sig = squeeze(X(:, ch, :));  % [epochs x samples]
            feats_ch = feature_extraction_extract_features(sig, CURRENT_ITERATION);
            feats_emg = [feats_emg, feats_ch]; %#ok<AGROW>
        end
        feats_all = [feats_all, feats_emg]; %#ok<AGROW>
        fprintf('  EMG features: %d channels -> %d features\n', nEMG, size(feats_emg,2));
    end

    features = feats_all;
    fprintf('Total multi-channel features: %d (per epoch)\n', size(features,2));
end
