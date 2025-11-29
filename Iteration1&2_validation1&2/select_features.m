function [features_sel, sel_idx] = select_features(features, labels)
% SELECT_FEATURES  Feature selection wrapper.
%
% Usage in main pipeline:
%   [selected_features, sel_idx] = select_features(features, labels);
%
% For Iteration 1  -> returns all features (no selection).
% For Iteration 2+ -> 
%   1) remove (near) constant features by variance
%   2) remove highly correlated features
%   3) rank remaining features and keep top-K (default K = 40)

    % Get CURRENT_ITERATION (default = 1 if not found)
    try
        CURRENT_ITERATION = evalin('base', 'CURRENT_ITERATION');
    catch
        CURRENT_ITERATION = 1;
    end

    % If Iteration 1: don't select, just pass through
    if CURRENT_ITERATION == 1
        features_sel = features;
        sel_idx = 1:size(features,2);
        fprintf('Feature selection (iter 1): using ALL %d features.\n', size(features,2));
        return;
    end

    fprintf('Feature selection (iteration %d)...\n', CURRENT_ITERATION);

    X = features;
    y = labels(:);

    nFeat0 = size(X,2);
    fprintf('  Start: %d features\n', nFeat0);

    %% 1) Variance thresholding (only remove constant / near-constant)
    var_feat = var(X, 0, 1);

    % keep features with variance above a tiny epsilon
    eps_var = 1e-10;
    keep_var = var_feat > eps_var;

    X1   = X(:, keep_var);
    idx1 = find(keep_var);
    fprintf('  After variance filter: %d features\n', size(X1,2));

    if isempty(X1)
        fprintf('  WARNING: variance filter removed all features -> keeping original.\n');
        features_sel = X;
        sel_idx      = 1:nFeat0;
        return;
    end

    %% 2) Correlation filter (remove highly correlated features)
    R = corr(X1);
    nF1 = size(X1, 2);
    to_drop = false(1, nF1);
    corr_thresh = 0.95;

    for i = 1:nF1
        if to_drop(i), continue; end
        for j = i+1:nF1
            if abs(R(i,j)) > corr_thresh && ~to_drop(j)
                to_drop(j) = true;
            end
        end
    end

    keep_corr = ~to_drop;
    X2        = X1(:, keep_corr);
    idx2      = idx1(keep_corr);

    fprintf('  After correlation filter: %d features\n', size(X2,2));

    if isempty(X2)
        fprintf('  WARNING: correlation filter removed all features -> keeping X1.\n');
        X2   = X1;
        idx2 = idx1;
    end

    %% 3) Ranking + select top-K
    % Target number of features (guide: 30–50, we pick 40 as default)
    K_TARGET = 40;

    nF2 = size(X2, 2);
    if nF2 <= K_TARGET
        % Not enough features to cut further
        features_sel = X2;
        sel_idx      = idx2;
        fprintf('  After ranking: %d features (no need to cut to %d)\n', nF2, K_TARGET);
    else
        % Prefer mutual information (fscmrmr), otherwise ANOVA F-score
        if exist('fscmrmr', 'file')
            % fscmrmr returns a ranking of feature indices (local to X2)
            rankIdxLocal = fscmrmr(X2, y);
        else
            fprintf('  fscmrmr not found -> using ANOVA-based ranking.\n');
            numF = size(X2,2);
            scores = zeros(1, numF);
            for f = 1:numF
                p = anova1(X2(:,f), y, 'off');
                scores(f) = -log10(p + eps);  % larger = more discriminative
            end
            [~, rankIdxLocal] = sort(scores, 'descend');
        end

        K = min(K_TARGET, numel(rankIdxLocal));
        sel_local = rankIdxLocal(1:K);

        features_sel = X2(:, sel_local);
        sel_idx      = idx2(sel_local);

        fprintf('  After ranking: selected top-%d -> %d features\n', K, size(features_sel,2));
    end

end
