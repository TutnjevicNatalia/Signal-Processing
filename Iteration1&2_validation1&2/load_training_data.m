function [varargout] = load_training_data(edfFolder, xmlFolder)
%% LOAD_TRAINING_DATA  Build 30 s epochs + stage labels from R1..R10.
%
%  Single-channel mode (Iteration 1):
%    [eeg_data, labels] = load_training_data(edfFolder, xmlFolder);
%    -> eeg_data : [nEpochs x nSamples] (combined C3+C4 EEG)
%
%  Multi-channel mode (Iteration 2+):
%    [mc_data, labels, ch_info] = load_training_data(edfFolder, xmlFolder);
%    -> mc_data.eeg : [nEpochs x nEEG x nSamplesEEG]
%       mc_data.eog : [nEpochs x nEOG x nSamplesEOG]
%       mc_data.emg : [nEpochs x nEMG x nSamplesEMG]
%
%  Labels:
%    labels : [nEpochs x 1], 0=Wake,1=N1,2=N2,3=N3,4=REM

    fprintf('Loading training data from %s (EDF) and %s (XML)...\n', ...
            edfFolder, xmlFolder);

    [hdr_all, record_all] = read_edf(edfFolder);
    nRecs = numel(hdr_all);

    recNames = {'R1','R2','R3','R4','R5','R6','R7','R8','R9','R10'};

    % Accumulators for single-channel (combined EEG) + labels
    eeg_epochs_all   = {};
    labels_all       = {};

    % Accumulators for multi-channel data (EEG/EOG/EMG)
    eeg_mc_all       = {};
    eog_mc_all       = {};
    emg_mc_all       = {};

    % NEW: per-epoch subject IDs (one vector per record)
    subject_ids_all  = {};   % {r} -> [nEpochs_valid_r x 1] with values r

    for r = 1:nRecs
        hdr    = hdr_all{r};
        record = record_all{r};
        recName = recNames{r};

        fprintf('\n--- Processing record %d (%s) ---\n', r, recName);

        labels_ch = string(hdr.label);

        % == Find channels (based on your actual labels) ==
        idx_C3    = find(labels_ch == "EEGsec", 1);  % C3-A2
        idx_C4    = find(labels_ch == "EEG",    1);  % C4-A1
        idx_EOG_L = find(labels_ch == "EOGL",   1);  % Left EOG
        idx_EOG_R = find(labels_ch == "EOGR",   1);  % Right EOG
        idx_EMG   = find(labels_ch == "EMG",    1);  % Chin EMG

        % === EEG (C3 + C4) ===
        if isempty(idx_C3) && isempty(idx_C4)
            warning('Record %d: no EEGsec/EEG channels found; skipping.', r);
            continue;
        end

        % List of EEG indices actually present
        eeg_idx = [idx_C3, idx_C4];
        eeg_idx = eeg_idx(~isnan(eeg_idx));  % scalar or empty; safe

        % Combine EEGs by averaging for single-channel mode
        if ~isempty(idx_C3) && ~isempty(idx_C4)
            eeg_signal_comb = (record(idx_C3,:) + record(idx_C4,:)) / 2;
            fprintf('  EEG: using AVERAGE of %s and %s\n', hdr.label{idx_C3}, hdr.label{idx_C4});
        else
            idx_single = eeg_idx(1);
            eeg_signal_comb = record(idx_single,:);
            fprintf('  EEG: using single channel %s\n', hdr.label{idx_single});
        end

        % EEG sampling (your dataset has 125 Hz for all these channels)
        fs_eeg  = double(hdr.samples(eeg_idx(1))) / double(hdr.duration);
        epochLenSec        = 30;
        samplesPerEpochEEG = round(fs_eeg * epochLenSec);

        % === EOG ===
        have_eog = ~isempty(idx_EOG_L) && ~isempty(idx_EOG_R);
        if have_eog
            fs_eog = double(hdr.samples(idx_EOG_L)) / double(hdr.duration);
            samplesPerEpochEOG = round(fs_eog * epochLenSec);
            fprintf('  EOG: using %s and %s\n', hdr.label{idx_EOG_L}, hdr.label{idx_EOG_R});
            eog_idx_all = [idx_EOG_L, idx_EOG_R];
        end

        % === EMG ===
        have_emg = ~isempty(idx_EMG);
        if have_emg
            fs_emg = double(hdr.samples(idx_EMG)) / double(hdr.duration);
            samplesPerEpochEMG = round(fs_emg * epochLenSec);
            fprintf('  EMG: using %s\n', hdr.label{idx_EMG});
        end

        % --- Stage labels from XML ---
        xmlFilename = fullfile(xmlFolder, [recName '.xml']);
        if ~exist(xmlFilename, 'file')
            warning('XML file not found for %s (%s); skipping.', recName, xmlFilename);
            continue;
        end

        [events, stages_sec, epochLength_xml, annotation] = readXML(xmlFilename);

        nEpochs_stages = floor(numel(stages_sec) / epochLenSec);
        fprintf('  Stage vector length = %d s -> %d epochs (stage-based)\n', ...
                numel(stages_sec), nEpochs_stages);

        % --- EEG: how many epochs from signal length? ---
        nSamplesEEG    = numel(eeg_signal_comb);
        nEpochs_signal = floor(nSamplesEEG / samplesPerEpochEEG);
        fprintf('  EEG signal length = %d samples -> %d epochs (signal-based)\n', ...
                nSamplesEEG, nEpochs_signal);

        nEpochs = min(nEpochs_signal, nEpochs_stages);
        if nEpochs == 0
            warning('Record %d: no complete epochs; skipping.', r);
            continue;
        end

        % --- Per-epoch labels (mode over 30 seconds) ---
        epoch_labels_orig = zeros(nEpochs, 1);
        for e = 1:nEpochs
            idx_start = (e-1)*epochLenSec + 1;
            idx_end   =  e   *epochLenSec;
            window    = stages_sec(idx_start:idx_end);
            epoch_labels_orig(e) = mode(window);
        end

        labels_epoch = zeros(nEpochs,1);
        for e = 1:nEpochs
            labels_epoch(e) = map_stage_code(epoch_labels_orig(e));
        end

        valid = ~isnan(labels_epoch);
        labels_epoch = labels_epoch(valid);
        nEpochs_valid = numel(labels_epoch);

        if nEpochs_valid == 0
            warning('Record %d: all epochs invalid after mapping; skipping.', r);
            continue;
        end

        % NEW: per-epoch subject IDs for this record (1..10)
        subject_ids_epoch = r * ones(nEpochs_valid, 1);

        % --- EEG epochs (combined C3+C4) ---
        eeg_epochs_comb = zeros(nEpochs_valid, samplesPerEpochEEG);
        e_count = 0;
        for e = 1:nEpochs
            if ~valid(e), continue; end
            e_count = e_count + 1;
            s_start = (e-1)*samplesPerEpochEEG + 1;
            s_end   =  e   *samplesPerEpochEEG;
            eeg_epochs_comb(e_count,:) = eeg_signal_comb(s_start:s_end);
        end

        % --- Multi-channel EEG (C3, C4 separately if both exist) ---
        if numel(eeg_idx) > 1
            nEEG = numel(eeg_idx);
            eeg_epochs_mc = zeros(nEpochs_valid, nEEG, samplesPerEpochEEG);
            for ch = 1:nEEG
                sig = record(eeg_idx(ch), :);
                e_count = 0;
                for e = 1:nEpochs
                    if ~valid(e), continue; end
                    e_count = e_count + 1;
                    s_start = (e-1)*samplesPerEpochEEG + 1;
                    s_end   =  e   *samplesPerEpochEEG;
                    eeg_epochs_mc(e_count, ch, :) = sig(s_start:s_end);
                end
            end
        else
            nEEG = 1;
            eeg_epochs_mc = zeros(nEpochs_valid, 1, samplesPerEpochEEG);
            sig = record(eeg_idx(1), :);
            e_count = 0;
            for e = 1:nEpochs
                if ~valid(e), continue; end
                e_count = e_count + 1;
                s_start = (e-1)*samplesPerEpochEEG + 1;
                s_end   =  e   *samplesPerEpochEEG;
                eeg_epochs_mc(e_count, 1, :) = sig(s_start:s_end);
            end
        end

        % --- EOG epochs ---
        if have_eog
            nEOG = 2;
            eog_epochs_mc = zeros(nEpochs_valid, nEOG, samplesPerEpochEOG);
            for ch = 1:nEOG
                sig = record(eog_idx_all(ch), :);  % use EOGL/EOGR indices
                nSamplesEOG        = numel(sig);
                nEpochs_signal_eog = floor(nSamplesEOG / samplesPerEpochEOG);
                nEpochs_ch         = min(nEpochs_signal_eog, nEpochs);
                e_count = 0;
                for e = 1:nEpochs_ch
                    if ~valid(e), continue; end
                    e_count = e_count + 1;
                    s_start = (e-1)*samplesPerEpochEOG + 1;
                    s_end   =  e   *samplesPerEpochEOG;
                    eog_epochs_mc(e_count, ch, :) = sig(s_start:s_end);
                end
            end
        else
            eog_epochs_mc = [];
        end

        % --- EMG epochs ---
        if have_emg
            nEMG = 1;
            emg_epochs_mc = zeros(nEpochs_valid, nEMG, samplesPerEpochEMG);
            sig = record(idx_EMG, :);
            nSamplesEMG        = numel(sig);
            nEpochs_signal_emg = floor(nSamplesEMG / samplesPerEpochEMG);
            nEpochs_ch         = min(nEpochs_signal_emg, nEpochs);
            e_count = 0;
            for e = 1:nEpochs_ch
                if ~valid(e), continue; end
                e_count = e_count + 1;
                s_start = (e-1)*samplesPerEpochEMG + 1;
                s_end   =  e   *samplesPerEpochEMG;
                emg_epochs_mc(e_count,1,:) = sig(s_start:s_end);
            end
        else
            emg_epochs_mc = [];
        end

        % --- Accumulate across records ---
        eeg_epochs_all{end+1}   = eeg_epochs_comb;   %#ok<AGROW>
        labels_all{end+1}       = labels_epoch;      %#ok<AGROW>
        subject_ids_all{end+1}  = subject_ids_epoch; %#ok<AGROW>

        if ~isempty(eeg_epochs_mc)
            eeg_mc_all{end+1} = eeg_epochs_mc; %#ok<AGROW>
        else
            eeg_mc_all{end+1} = [];
        end

        if ~isempty(eog_epochs_mc)
            eog_mc_all{end+1} = eog_epochs_mc; %#ok<AGROW>
        else
            eog_mc_all{end+1} = [];
        end

        if ~isempty(emg_epochs_mc)
            emg_mc_all{end+1} = emg_epochs_mc; %#ok<AGROW>
        else
            emg_mc_all{end+1} = [];
        end
    end

    % --- Concatenate across all records (single-channel) ---
    eeg_data    = cat(1, eeg_epochs_all{:});
    labels      = cat(1, labels_all{:});
    subject_ids = cat(1, subject_ids_all{:});   % NEW: [nEpochs x 1], values 1..10

    fprintf('\nFinished building training data.\n');
    fprintf('  Total epochs: %d\n', size(eeg_data,1));
    fprintf('  Samples per epoch (EEG): %d\n', size(eeg_data,2));

    % === Output mode ===
    if nargout <= 2
        % Single-channel mode (Iteration 1)
        varargout{1} = eeg_data;
        varargout{2} = labels;

    else
        % Multi-channel mode (Iteration 2+)
        multi_channel_data = struct();

        if any(~cellfun(@isempty, eeg_mc_all))
            multi_channel_data.eeg = cat(1, eeg_mc_all{:});
        end
        if any(~cellfun(@isempty, eog_mc_all))
            multi_channel_data.eog = cat(1, eog_mc_all{:});
        end
        if any(~cellfun(@isempty, emg_mc_all))
            multi_channel_data.emg = cat(1, emg_mc_all{:});
        end

        % Channel info (using last fs_* seen; they should be constant)
        channel_info = struct();
        if isfield(multi_channel_data,'eeg')
            channel_info.eeg.names = {};   % you can fill with actual labels if you want
            channel_info.eeg.fs    = fs_eeg;
        end
        if exist('fs_eog','var') && isfield(multi_channel_data,'eog')
            channel_info.eog.names = {};
            channel_info.eog.fs    = fs_eog;
        end
        if exist('fs_emg','var') && isfield(multi_channel_data,'emg')
            channel_info.emg.names = {};
            channel_info.emg.fs    = fs_emg;
        end

        % NEW: subject info for LOSO
        channel_info.subject_ids   = subject_ids;   % [nEpochs x 1], 1..10
        channel_info.subject_names = recNames;      % {'R1',...,'R10'}

        varargout{1} = multi_channel_data;
        varargout{2} = labels;
        varargout{3} = channel_info;
    end
end

function y = map_stage_code(x)
    % 0 = REM, 1 = N4, 2 = N3, 3 = N2, 4 = N1, 5 = Wake
    switch x
        case 5, y = 0;           % Wake
        case 4, y = 1;           % N1
        case 3, y = 2;           % N2
        case {2,1}, y = 3;       % N3+N4
        case 0, y = 4;           % REM
        otherwise, y = NaN;
    end
end
