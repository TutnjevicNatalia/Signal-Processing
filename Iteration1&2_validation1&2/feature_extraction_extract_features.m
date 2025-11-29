function features = feature_extraction_extract_features(data, CURRENT_ITERATION)
%% FEATURE_EXTRACTION_EXTRACT_FEATURES
% Time-domain + (for iteration >= 2) Welch spectral features.
%
% Input:
%   data             : [nEpochs x nSamples]
%   CURRENT_ITERATION: scalar (1,2,...)
%
% Output:
%   features         : [nEpochs x nFeatures]

    if nargin < 2
        CURRENT_ITERATION = 1;
    end

    % Get sampling rate (assuming 125 Hz for this dataset)
    try
        fs = evalin('base', 'FS_EEG');
    catch
        fs = 125;
    end

    n_epochs = size(data, 1);

    %% --- 16 time-domain features (used in all iterations) ---
    % 1)  mean
    % 2)  median
    % 3)  std
    % 4)  var
    % 5)  rms
    % 6)  min
    % 7)  max
    % 8)  range
    % 9)  skewness
    % 10) kurtosis
    % 11) zero-crossings
    % 12) Hjorth activity
    % 13) Hjorth mobility
    % 14) Hjorth complexity
    % 15) energy
    % 16) power

    n_time_feats = 16;
    time_feats = zeros(n_epochs, n_time_feats);

    for i = 1:n_epochs
        epoch = double(data(i, :));

        mu    = mean(epoch);
        med   = median(epoch);
        sd    = std(epoch);
        vr    = var(epoch);
        rmsv  = rms(epoch);
        mn    = min(epoch);
        mx    = max(epoch);
        rg    = mx - mn;
        sk    = skewness(epoch);
        kt    = kurtosis(epoch);

        % Zero-crossings
        zc = sum(diff(sign(epoch)) ~= 0);

        % Hjorth parameters
        diff1 = diff(epoch);
        diff2 = diff(diff1);

        var0 = vr;                  % variance of signal
        var1 = var(diff1);          % variance of first derivative
        var2 = var(diff2);          % variance of second derivative

        hj_activity   = var0;
        hj_mobility   = sqrt(var1 / (var0 + eps));
        hj_complexity = sqrt((var2 / (var1 + eps))) / (hj_mobility + eps);

        energy = sum(epoch.^2);
        power  = mean(epoch.^2);

        time_feats(i, :) = [ ...
            mu, med, sd, vr, rmsv, mn, mx, rg, ...
            sk, kt, zc, ...
            hj_activity, hj_mobility, hj_complexity, ...
            energy, power ...
        ];
    end

    % --- Iteration 1: only time-domain ---
    if CURRENT_ITERATION == 1
        features = time_feats;
        fprintf('Extracted %d features from %d epochs (time-domain only)\n', ...
                size(features,2), n_epochs);
        return;
    end

    %% --- Iteration 2+: add Welch spectral features ---
    % We add:
    %  - 5 absolute band powers:   delta, theta, alpha, sigma, beta
    %  - 5 relative band powers:   normalized by total power (0.5–40 Hz)
    %  - 1 spectral entropy
    %  - 1 peak frequency
    %  - 1 spectral edge frequency (95% cumulative power)
    %
    % => 13 spectral features per epoch

    n_spec_feats = 13;   % IMPORTANT: must match what compute_spectral_features_welch returns
    spec_feats = zeros(n_epochs, n_spec_feats);

    for i = 1:n_epochs
        epoch = double(data(i, :));
        spec_feats(i, :) = compute_spectral_features_welch(epoch, fs);
    end

    features = [time_feats, spec_feats];

    fprintf('Extracted %d features from %d epochs (16 time + %d spectral)\n', ...
            size(features,2), n_epochs, n_spec_feats);
end


%% === Helper: Welch spectral features for one epoch ======================
function f = compute_spectral_features_welch(epoch, fs)
% Compute 13 spectral features from a single epoch using Welch's method.
%
% Output vector f (1 x 13):
%   [ 5 abs band powers,
%     5 relative band powers,
%     1 spectral entropy,
%     1 peak frequency,
%     1 spectral edge frequency (95%) ]

    epoch = epoch(:);   % column vector

    % --- Welch parameters ---
    % 4-second Hann window, 50% overlap, nfft >= 512
    win_len = round(4 * fs);
    if win_len > numel(epoch)
        win_len = numel(epoch);
    end

    win      = hann(win_len);
    noverlap = floor(0.5 * win_len);
    nfft     = max(512, 2^nextpow2(numel(epoch)));

    [psd, freqs] = pwelch(epoch, win, noverlap, nfft, fs);

    % Focus on 0.5–40 Hz range
    idx = (freqs >= 0.5) & (freqs <= 40);
    freqs = freqs(idx);
    psd   = psd(idx);

    if numel(freqs) < 2
        % Degenerate case – return zeros
        f = zeros(1,13);
        return;
    end

    df = mean(diff(freqs));               % frequency resolution
    total_power = sum(psd) * df;          % total power in 0.5–40 Hz

    % --- Define classical EEG bands ---
    %   delta: 0.5–4 Hz
    %   theta: 4–8 Hz
    %   alpha: 8–12 Hz
    %   sigma: 12–15 Hz
    %   beta : 15–30 Hz
    bands = [ ...
        0.5   4;   % delta
        4     8;   % theta
        8    12;   % alpha
        12   15;   % sigma
        15   30];  % beta

    nb = size(bands,1);
    absP = zeros(1, nb);

    for b = 1:nb
        bi = (freqs >= bands(b,1)) & (freqs < bands(b,2));
        absP(b) = sum(psd(bi)) * df;
    end

    relP = absP / (total_power + eps);

    % --- Spectral entropy ---
    p_norm = psd / (sum(psd) + eps);
    spec_entropy = -sum(p_norm .* log(p_norm + eps));

    % --- Peak frequency ---
    [~, imax] = max(psd);
    peak_f = freqs(imax);

    % --- Spectral edge frequency (95% of cumulative power) ---
    cumsumP = cumsum(psd) * df;
    edge_idx = find(cumsumP >= 0.95 * total_power, 1, 'first');
    if isempty(edge_idx)
        edge_f = freqs(end);
    else
        edge_f = freqs(edge_idx);
    end

    % Pack features: 5 absP + 5 relP + entropy + peak_f + edge_f = 13
    f = [absP, relP, spec_entropy, peak_f, edge_f];
end
