%% check_epoch_counts.m
% Compares EDF-based epoch counts vs XML-based epoch counts for R1–R10

clear; clc;

run('config.m');   % Make sure TRAINING_DIR is defined

edf_dir = TRAINING_DIR;
xml_dir = TRAINING_DIR;

records = dir(fullfile(edf_dir, 'R*.edf'));

if isempty(records)
    error('No R*.edf files found in %s', edf_dir);
end

epoch_table = [];

fprintf('\n==================== Checking EDF vs XML epoch counts ====================\n');

for k = 1:length(records)

    edfName = records(k).name;
    recID   = erase(edfName, '.edf');

    fprintf('\n--- Processing %s ---\n', recID);

    %% ===== Read EDF =====
    edfPath = fullfile(edf_dir, edfName);
    [hdr, record] = edfread(edfPath);
    
    % Use EEGsec (C3-A2) or EEG if not available
    labels = string(hdr.label);
    idx_C3 = find(labels == "EEGsec", 1);
    idx_C4 = find(labels == "EEG",    1);

    eeg_idx = [idx_C3, idx_C4];
    eeg_idx = eeg_idx(~isnan(eeg_idx));

    if isempty(eeg_idx)
        warning('%s: No EEG channels found, skipping.', recID);
        continue;
    end

    fs = double(hdr.samples(eeg_idx(1))) / double(hdr.duration);
    samplesPerEpoch = round(fs * 30);

    % Use first EEG channel for epoch count
    eeg_sig = record(eeg_idx(1), :);
    nSamples = numel(eeg_sig);

    edf_epochs = floor(nSamples / samplesPerEpoch);
    fprintf('  EDF-based epochs: %d\n', edf_epochs);


    %% ===== Read XML =====
    xmlPath = fullfile(xml_dir, [recID '.xml']);
    if ~exist(xmlPath, 'file')
        error('XML file not found: %s', xmlPath);
    end

    try
        [events, stages_sec, epochLength_xml, annotation] = readXML(xmlPath);
    catch ME
        error('Error reading XML %s: %s', xmlPath, ME.message);
    end

    % length(stages_sec) = one value per second
    xml_seconds = length(stages_sec);
    xml_epochs = floor(xml_seconds / 30);

    fprintf('  XML-based epochs: %d\n', xml_epochs);

    %% ===== Store results =====
    epoch_table = [epoch_table; {recID, edf_epochs, xml_epochs, edf_epochs - xml_epochs}];

end

%% ===== Print Summary Table =====
fprintf('\n==================== SUMMARY ====================\n');

T = cell2table(epoch_table, ...
    'VariableNames', {'Record','EDF_Epochs','XML_Epochs','Difference_EDF_minus_XML'});

disp(T);

fprintf('\nIf every record shows Difference = 1 → You MUST drop 1 epoch per H-file.\n');
