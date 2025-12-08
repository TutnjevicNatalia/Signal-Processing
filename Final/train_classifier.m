function model = train_classifier(features, labels)
% TRAIN_CLASSIFIER  Wrapper for kNN / SVM / Random Forest.
%
% Uses CLASSIFIER_TYPE from base workspace:
%   'knn'           -> k-NN (Iteration 1 default)
%   'svm'           -> SVM via fitcecoc (Iteration 2)
%   'random_forest' -> Random Forest via fitcensemble('Bag') (Iteration 3)
%
% Other base vars used (if present):
%   KNN_K
%   SVM_KERNEL, SVM_C, SVM_KERNEL_SCALE
%   RF_NUM_TREES, RF_MAX_DEPTH, RF_MIN_LEAF

    X = features;
    y = labels(:);

    % ---- Read iteration & classifier type ----
    try
        CURRENT_ITERATION = evalin('base', 'CURRENT_ITERATION');
    catch
        CURRENT_ITERATION = 1;
    end

    try
        CLASSIFIER_TYPE = evalin('base', 'CLASSIFIER_TYPE');
    catch
        % sensible defaults per iteration
        if CURRENT_ITERATION == 1
            CLASSIFIER_TYPE = 'knn';
        elseif CURRENT_ITERATION == 2
            CLASSIFIER_TYPE = 'svm';
        else
            CLASSIFIER_TYPE = 'random_forest';
        end
    end

    CLASSIFIER_TYPE_L = lower(strtrim(CLASSIFIER_TYPE));

    switch CLASSIFIER_TYPE_L

        %% ----------------------------------------------------------------
        %  k-NN (Iteration 1)
        % -----------------------------------------------------------------
        case 'knn'
            k = 5;
            try
                k = evalin('base','KNN_K');
            catch
            end

            fprintf('Training knn classifier for iteration %d (k=%d)...\n', ...
                CURRENT_ITERATION, k);

            model = fitcknn(X, y, ...
                'NumNeighbors', k, ...
                'Standardize', true);

        %% ----------------------------------------------------------------
        %  SVM via ECOC (Iteration 2)
        % -----------------------------------------------------------------
        case 'svm'
            % defaults
            svm_kernel       = 'rbf';
            svm_C            = 1.0;
            svm_kernel_scale = 'auto';

            % override from base if available
            try, svm_kernel       = evalin('base','SVM_KERNEL');       end
            try, svm_C            = evalin('base','SVM_C');            end
            try, svm_kernel_scale = evalin('base','SVM_KERNEL_SCALE'); end

            fprintf('Training svm classifier for iteration %d...\n', CURRENT_ITERATION);

            t = templateSVM( ...
                'KernelFunction', lower(svm_kernel), ...
                'BoxConstraint',  svm_C, ...
                'KernelScale',    svm_kernel_scale, ...
                'Standardize',    true);

            model = fitcecoc(X, y, ...
                'Learners', t, ...
                'Coding',   'onevsall', ...
                'ClassNames', unique(y));

        %% ----------------------------------------------------------------
        %  Random Forest (Iteration 3) - bagged trees
        % -----------------------------------------------------------------
        case {'random_forest','rf','randomforest'}
            % hyperparams
            nTrees     = 200;
            maxDepth   = 20;
            minLeaf    = 5;

            try, nTrees   = evalin('base','RF_NUM_TREES');   end
            try, maxDepth = evalin('base','RF_MAX_DEPTH');   end
            try, minLeaf  = evalin('base','RF_MIN_LEAF');    end

            fprintf('Training random_forest classifier for iteration %d...\n', ...
                CURRENT_ITERATION);
            fprintf('  #Trees=%d, MaxDepth=%d, MinLeaf=%d (class_weight=balanced)\n', ...
                nTrees, maxDepth, minLeaf);

            % --- class weighting (balanced) ---
            classes = unique(y);
            counts  = arrayfun(@(c)sum(y==c), classes);
            w_per_class = 1 ./ counts;   % inverse frequency
            w = zeros(size(y));
            for i = 1:numel(classes)
                w(y == classes(i)) = w_per_class(i);
            end

            % Approximate depth via MaxNumSplits (2^depth - 1)
            % We'll just pass maxDepth as MaxNumSplits (reasonable heuristic)
            template = templateTree( ...
                'MaxNumSplits', maxDepth, ...
                'MinLeafSize',  minLeaf);

            model = fitcensemble(X, y, ...
                'Method',            'Bag', ...
                'Learners',          template, ...
                'NumLearningCycles', nTrees, ...
                'Weights',           w, ...
                'ClassNames',        classes);

        %% ----------------------------------------------------------------
        %  Fallback
        % -----------------------------------------------------------------
        otherwise
            warning('Unknown CLASSIFIER_TYPE "%s", falling back to k-NN (k=5).', ...
                CLASSIFIER_TYPE);
            k = 5;
            fprintf('Training knn classifier (fallback)...\n');
            model = fitcknn(X, y, ...
                'NumNeighbors', k, ...
                'Standardize', true);
    end
end