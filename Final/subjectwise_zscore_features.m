function Xz = subjectwise_zscore_features(X, subject_ids)
% SUBJECTWISE_ZSCORE_FEATURES
% Z-score features within each subject separately.
%
% X          : [nEpochs x nFeatures]
% subject_ids: [nEpochs x 1] (e.g. 1..10 for R1..R10)
%
% Output:
%   Xz: same size as X, each subject has mean=0, std=1 per feature.

    X = double(X);
    subject_ids = subject_ids(:);
    Xz = X;

    uSubs = unique(subject_ids(~isnan(subject_ids)));
    for k = 1:numel(uSubs)
        s = uSubs(k);
        idx = (subject_ids == s);
        if ~any(idx), continue; end

        Xs = X(idx, :);
        mu_s = mean(Xs, 1);
        sig_s = std(Xs, 0, 1) + 1e-6;

        Xz(idx, :) = (Xs - mu_s) ./ sig_s;
    end
end
