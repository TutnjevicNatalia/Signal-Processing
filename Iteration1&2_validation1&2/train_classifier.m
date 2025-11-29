function model = train_classifier(features, labels)
%% TRAIN_CLASSIFIER  Train classifier based on config.m (iteration-aware).
%
% Uses config variables from the *base* workspace:
%   CURRENT_ITERATION, CLASSIFIER_TYPE, KNN_N_NEIGHBORS,
%   SVM_KERNEL, SVM_C, SVM_KERNEL_SCALE

    % --- Read config from base workspace ---
    try
        CURRENT_ITERATION = evalin('base', 'CURRENT_ITERATION');
    catch
        CURRENT_ITERATION = 1;
    end

    try
        CLASSIFIER_TYPE = evalin('base', 'CLASSIFIER_TYPE');
    catch
        % Fallback based on iteration
        if CURRENT_ITERATION == 1
            CLASSIFIER_TYPE = 'knn';
        else
            CLASSIFIER_TYPE = 'svm';
        end
    end

    % For k-NN params
    try
        KNN_N_NEIGHBORS = evalin('base', 'KNN_N_NEIGHBORS');
    catch
        KNN_N_NEIGHBORS = 5;
    end

    % For SVM params (with defaults)
    svm_kernel      = 'rbf';
    svm_C           = 1.0;
    svm_kernelScale = 'auto';
    try, svm_kernel      = evalin('base', 'SVM_KERNEL');        end
    try, svm_C           = evalin('base', 'SVM_C');             end
    try, svm_kernelScale = evalin('base', 'SVM_KERNEL_SCALE');  end

    fprintf('Training %s classifier for iteration %d...\n', CLASSIFIER_TYPE, CURRENT_ITERATION);

    % --- Train model based on CLASSIFIER_TYPE ---
    switch lower(CLASSIFIER_TYPE)

        case 'knn'
            % k-NN classifier
            model = fitcknn(features, labels, ...
                            'NumNeighbors', KNN_N_NEIGHBORS);
            fprintf('k-NN classifier trained with k=%d\n', KNN_N_NEIGHBORS);

        case 'svm'
            % Multi-class SVM via ECOC (one-vs-all or one-vs-one)
            t = templateSVM( ...
                    'KernelFunction', svm_kernel, ...
                    'BoxConstraint',  svm_C, ...
                    'KernelScale',    svm_kernelScale);

            % One-vs-all coding for multi-class
            model = fitcecoc(features, labels, ...
                             'Learners', t, ...
                             'Coding',   'onevsall');

            fprintf('Multi-class SVM (ECOC) trained (kernel=%s, C=%.3g, kernelScale=%s)\n', ...
                    svm_kernel, svm_C, mat2str(svm_kernelScale));

        otherwise
            warning('Unknown CLASSIFIER_TYPE "%s", falling back to k-NN (k=5).', CLASSIFIER_TYPE);
            model = fitcknn(features, labels, 'NumNeighbors', 5);
    end

    % --- Evaluate on training data (for demonstration only) ---
    predictions = predict(model, features);
    predictions = predictions(:);
    labels      = labels(:);

    accuracy = sum(predictions == labels) / numel(labels) * 100;
    fprintf('Training accuracy: %.2f%%\n', accuracy);
end
