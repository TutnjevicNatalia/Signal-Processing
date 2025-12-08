function feature_names = build_feature_names_multichannel(preprocessed_data, CURRENT_ITERATION)
% BUILD_FEATURE_NAMES_MULTICHANNEL
% Construct human-readable names for each column in the feature matrix
% produced by extract_features_multichannel, in the SAME order.

if nargin < 2 || isempty(CURRENT_ITERATION)
    try
        CURRENT_ITERATION = evalin('base','CURRENT_ITERATION');
    catch
        CURRENT_ITERATION = 2;
    end
end

feature_names = {};

%% ----- Base (29) names from feature_extraction_extract_features -----
% These MUST match the order in feature_extraction_extract_features.m

time_feat_names = { ...
    'mean', ...
    'median', ...
    'std', ...
    'var', ...
    'rms', ...
    'min', ...
    'max', ...
    'range', ...
    'skewness', ...
    'kurtosis', ...
    'zero_crossings', ...
    'hjorth_activity', ...
    'hjorth_mobility', ...
    'hjorth_complexity', ...
    'energy', ...
    'power'};

spec_band_labels = {'delta', 'theta', 'alpha', 'sigma', 'beta'};
spec_abs_names = strcat('abs_', spec_band_labels);
spec_rel_names = strcat('rel_', spec_band_labels);
spec_other_names = {'spec_entropy', 'peak_freq', 'edge_freq'};

spec_feat_names = [spec_abs_names, spec_rel_names, spec_other_names];

base29_names = [time_feat_names, spec_feat_names];  % 16 + 13

%% ----- Wavelet names (20 total from compute_wavelet_features_basic) -----
% Order in feats(i,:) = [E, RMS, LE, ENT] over bands {A4, D4, D3, D2, D1}

bands      = {'A4','D4','D3','D2','D1'};
stats      = {'E','RMS','logE','ENT'};
wavelet20_names = cell(1, 20);
idx = 1;
for s = 1:numel(stats)
    for b = 1:numel(bands)
        wavelet20_names{idx} = sprintf('wav_%s_%s', bands{b}, stats{s});
        idx = idx + 1;
    end
end
% For EOG/EMG, you only keep first 10 -> E & RMS across 5 bands
wavelet10_names = wavelet20_names(1:10);

%% ==================== EEG ====================
if isfield(preprocessed_data,'eeg') && ~isempty(preprocessed_data.eeg)
    [~, nEEG, ~] = size(preprocessed_data.eeg);

    for ch = 1:nEEG
        prefix = sprintf('EEG%d_', ch);

        if CURRENT_ITERATION >= 3
            % 29 base + 20 wavelet
            names_ch = [ ...
                strcat(prefix, base29_names), ...
                strcat(prefix, wavelet20_names) ...
            ];
        else
            % only 29 base
            names_ch = strcat(prefix, base29_names);
        end

        feature_names = [feature_names, names_ch]; %#ok<AGROW>
    end
end

%% ==================== EOG ====================
if isfield(preprocessed_data,'eog') && ~isempty(preprocessed_data.eog)
    [~, nEOG, ~] = size(preprocessed_data.eog);

    for ch = 1:nEOG
        prefix = sprintf('EOG%d_', ch);

        if CURRENT_ITERATION >= 3
            % 29 base + 10 wavelet (first 10 from wavelet20_names)
            names_ch = [ ...
                strcat(prefix, base29_names), ...
                strcat(prefix, wavelet10_names) ...
            ];
        else
            % Iter 2: 29 base only
            names_ch = strcat(prefix, base29_names);
        end

        feature_names = [feature_names, names_ch]; %#ok<AGROW>
    end
end

%% ==================== EMG (iter >= 3) ====================
if isfield(preprocessed_data,'emg') && ~isempty(preprocessed_data.emg) && CURRENT_ITERATION >= 3
    [~, nEMG, ~] = size(preprocessed_data.emg);

    base_emg_names = {'total_power', 'var', 'hf_power'};  % from compute_emg_features

    for ch = 1:nEMG
        prefix = sprintf('EMG%d_', ch);
        names_ch = [ ...
            strcat(prefix, base_emg_names), ...
            strcat(prefix, wavelet10_names) ...
        ];
        feature_names = [feature_names, names_ch]; %#ok<AGROW>
    end
end

feature_names = feature_names(:);  % column cell array
fprintf('Built %d feature names for multichannel features.\n', numel(feature_names));
end
