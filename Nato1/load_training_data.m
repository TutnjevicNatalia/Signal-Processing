function [eeg_data, labels] = load_training_data(edfFilePath, xmlFilePath)
%% STUDENT IMPLEMENTATION AREA: Load EDF and XML files.
%
% This function currently returns DUMMY DATA for jumpstart testing.
% Students must implement actual EDF/XML loading:
%
% 1. Load EDF file using read_edf function
% 2. Load XML annotations (sleep stage labels)
% 3. Extract relevant channels (EEG, EOG, EMG)
% 4. Segment into 30-second epochs
% 5. Handle different sampling rates
% 6. Match epochs with sleep stage labels

fprintf('Loading training data from %s and %s...\n', edfFilePath, xmlFilePath);


%[hdr, record] = read_edf(filePath); 

% TODO: Students must implement actual file loading
% DUMMY DATA for jumpstart testing - students must replace this:
fprintf('WARNING: Using dummy data! Students must implement actual EDF/XML loading.\n');

% NOTE FOR STUDENTS: This study has specific sampling rates:
% - EEG (C3-A2, C4-A1): 125 Hz
% - EOG (Left, Right): 50 Hz
% - EMG: 125 Hz
% Students must handle different sampling rates when loading real EDF files


%% LOAD_TRAINING_DATA  Build 30 s EEG epochs + stage labels from R1..R10.
%
%   [eeg_data, labels] = load_training_data(dataFolder, ~)
%
%   Inputs:
%       edfFilePath : folder that contains R1..R10 .edf and .xml files
%       xmlFilePath : (unused here; kept for compatibility)
%
%   Outputs:
%       eeg_data : [nEpochsTotal x nSamplesPerEpoch] EEG matrix
%       labels   : [nEpochsTotal x 1] integer labels:
%                  0 = Wake, 1 = N1, 2 = N2, 3 = N3, 4 = REM
%
%   Uses your read_edf(filePath) which returns:
%       hdr_all{r}, record_all{r} for r = 1..10 (R1..R10)

    fprintf('Loading EDFs from folder: %s\n', edfFilePath);

    % 1) Load all 10 EDFs: R1..R10
    [hdr_all, record_all] = read_edf(edfFilePath);
    nRecs = numel(hdr_all);

    % We will accumulate epochs and labels from all records
    eeg_epochs_all = {}; % 1 
    labels_all     = {};

    % recNames must match read_edf
    recNames = {'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10'};

    for r = 1:nRecs
        hdr    = hdr_all{r};     % header struct for Rr
        record = record_all{r};  % [nCh x nSamples] for Rr

        recName = recNames{r};
        fprintf('\n--- Processing record %d (%s) ---\n', r, recName);

        % 2) Find EEG channel (C3-A2: EEGsec, or fallback to EEG)
        labels_ch = string(hdr.label);

        idx_C3 = find(labels_ch == "EEGsec", 1);    % Linked C3-A2
        idx_C4 = find(labels_ch == "EEG",    1);    % Linked C4-A1 (as you showed)

        if isempty(idx_C3) && isempty(idx_C4)
            warning('Record %d: no EEGsec/EEG channel found; skipping.', r);
            continue;
        end

        if ~isempty(idx_C3)
            eeg_idx  = idx_C3;
        else
            eeg_idx  = idx_C4;
        end
        eeg_name = hdr.label{eeg_idx};
        fprintf('  Using EEG channel %d (%s)\n', eeg_idx, eeg_name);

        % 3) EEG sampling frequency and epoch length in samples
        fs_eeg  = double(hdr.samples(eeg_idx)) / double(hdr.duration);  % Hz
        epochLenSec = 30;   % we will use 30 s epochs for EEG
        samplesPerEpoch = round(fs_eeg * epochLenSec);

        eeg_signal = record(eeg_idx, :);        % 1 x nSamples
        nSamples   = numel(eeg_signal);
        nEpochs_signal = floor(nSamples / samplesPerEpoch);

        fprintf('  fs EEG = %.2f Hz, epoch length = %d s -> %d samples/epoch\n', ...
                fs_eeg, epochLenSec, samplesPerEpoch);
        fprintf('  EEG signal length = %d samples -> %d epochs (signal-based)\n', ...
                nSamples, nEpochs_signal);

        % 4) Load XML for this record and get stages
        xmlFilename = fullfile(edfFilePath, [recName '.xml']);
        if ~exist(xmlFilename, 'file')
            warning('XML file not found for %s (%s); skipping.', recName, xmlFilename);
            continue;
        end

        [events, stages_sec, epochLength_xml, annotation] = readXML(xmlFilename);

        % Your readXML builds "stages" one value per SECOND, with:
        %   0 = REM, 1 = N4, 2 = N3, 3 = N2, 4 = N1, 5 = Wake
        % Epoch length in seconds:
        fprintf('  XML epoch length = %d s\n', epochLength_xml);

        % We'll derive labels per 30 s epoch using the mode of 30 seconds
        nEpochs_stages = floor(numel(stages_sec) / epochLenSec);
        fprintf('  Stage vector length = %d s -> %d epochs (stage-based)\n', ...
                numel(stages_sec), nEpochs_stages);

        % Use the minimum to keep them aligned
        nEpochs = min(nEpochs_signal, nEpochs_stages);
        if nEpochs == 0
            warning('Record %d: no complete epochs; skipping.', r);
            continue;
        end

        % 5) Convert per-second stages to per-epoch labels using mode
        epoch_labels_orig = zeros(nEpochs, 1);  % original 0..5 coding
        for e = 1:nEpochs
            idx_start = (e-1)*epochLenSec + 1;
            idx_end   =  e   *epochLenSec;
            window    = stages_sec(idx_start:idx_end); % extract the intervals 
            epoch_labels_orig(e) = mode(window);
        end

        % 6) Map original codes (0..5) -> labels 0..4 (Wake,N1,N2,N3,REM)
        labels_epoch = zeros(nEpochs, 1);
        for e = 1:nEpochs
            labels_epoch(e) = map_stage_code(epoch_labels_orig(e));
        end

        % Drop invalid labels (NaN), if any
        valid = ~isnan(labels_epoch);
        labels_epoch = labels_epoch(valid);
        nEpochs_valid = numel(labels_epoch);

        if nEpochs_valid == 0
            warning('Record %d: all epochs invalid after mapping; skipping.', r);
            continue;
        end

        % 7) Segment EEG into 30 s epochs aligned with labels
        eeg_epochs = zeros(nEpochs_valid, samplesPerEpoch);
        e_count = 0;
        for e = 1:nEpochs
            if ~valid(e), continue; end %skip invalid epochs 
            e_count = e_count + 1;

            s_start = (e-1)*samplesPerEpoch + 1;
            s_end   =  e   *samplesPerEpoch;
            eeg_epochs(e_count, :) = eeg_signal(s_start:s_end);
        end

        % Store for this record
        eeg_epochs_all{end+1} = eeg_epochs; %#ok<AGROW>
        labels_all{end+1}     = labels_epoch; %#ok<AGROW>
    end

    % 8) Concatenate across all R1..R10
    if isempty(eeg_epochs_all)
        error('No valid epochs found in any record.');
    end

    eeg_data = cat(1, eeg_epochs_all{:});
    labels   = cat(1, labels_all{:});

    fprintf('\nFinished building training data.\n');
    fprintf('  Total epochs: %d\n', size(eeg_data,1));
    fprintf('  Samples per epoch: %d\n', size(eeg_data,2));

    stage_names = {'Wake','N1','N2','N3','REM'};
    for s = 0:4
        cnt = sum(labels == s);
        fprintf('  %s: %d epochs (%.1f%%)\n', stage_names{s+1}, cnt, 100*cnt/numel(labels));
    end

end


% ---------- Helper: map original stage code (0..5) to 0..4 ----------
function y = map_stage_code(x)
    % From your readXML:
    %   0 = REM, 1 = N4, 2 = N3, 3 = N2, 4 = N1, 5 = Wake
    % Desired labels:
    %   0 = Wake, 1 = N1, 2 = N2, 3 = N3 (N3 + N4), 4 = REM

    switch x
        case 5         % Wake
            y = 0;
        case 4         % N1
            y = 1;
        case 3         % N2
            y = 2;
        case {2, 1}    % N3 and N4 -> N3
            y = 3;
        case 0         % REM
            y = 4;
        otherwise
            y = NaN;   % unknown / ignore
    end
end