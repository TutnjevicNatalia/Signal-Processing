function features = extract_features(data)
%% Extract features - wrapper that accesses config from BASE workspace

    try
        % Read CURRENT_ITERATION from base workspace (config.m)
        CURRENT_ITERATION = evalin('base', 'CURRENT_ITERATION');
    catch
        CURRENT_ITERATION = 1; % Default to iteration 1
    end

    fprintf('Extracting features for iteration %d...\n', CURRENT_ITERATION);

    % Call the actual implementation
    features = feature_extraction_extract_features(data, CURRENT_ITERATION);
end