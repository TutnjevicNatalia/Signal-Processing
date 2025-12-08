function [multi_channel_data, channel_info, record_numbers, epoch_numbers] = load_holdout_data(holdoutFolder)
%% LOAD_HOLDOUT_DATA  Build 30 s epochs from H1..H* EDFs (NO labels).
%
% Output:
%   multi_channel_data.eeg : [nEpochs x nEEG x nSamplesEEG]
%   multi_channel_data.eog : [nEpochs x nEOG x nSamplesEOG] (if present)
%   multi_channel_data.emg : [nEpochs x nEMG x nSamplesEMG] (if present)
%
%   channel_info.eeg.fs, .eog.fs, .emg.fs
%   channel_info.subject_ids   : [nEpochs x 1], 1..nRecs
%   channel_info.subject_names : {'H1','H2', ...}
%
%   record_numbers : [nEpochs x 1] string array like "H1","H1","H2",...
%   epoch_numbers  : [nEpochs x 1] epoch indices within each recording (1,2,...)

fprintf('Loading HOLDOUT data from %s (EDF only).\n', holdoutFolder);

% Find all H*.edf files in the folder
edfFiles = dir(fullfile(holdoutFolder, 'H*.edf'));
if isempty(edfFiles)
    error('No H*.edf files found in %s', holdoutFolder);
end

nRecs = numel(edfFiles);

% Accumulators for multi-channel data
eeg_mc_all = {};
eog_mc_all = {};
emg_mc_all = {};

% For mapping epochs back to (record, epoch)
record_numbers_all = {};
epoch_numbers_all  = {};
subject_ids_all    = {};
subject_names      = cell(1, nRecs);

% Some variables to fill channel_info later
fs_eeg = [];
fs_eog = [];
fs_emg = [];

epochLenSec = 30;  % 30-second epochs (same as training)

for r = 1:nRecs
    edfName = edfFiles(r).name;          % e.g. 'H1.edf'
    recName = erase(edfName, '.edf');    % e.g. 'H1'
    subject_names{r} = recName;

    fprintf('\n--- Processing holdout record %d (%s) ---\n', r, recName);

    edfPath = fullfile(holdoutFolder, edfName);
    [hdr, record] = edfread(edfPath);
    record = double(record);

    labels_ch = string(hdr.label);

    % === Find channels (same labels as training) ===
    idx_C3    = find(labels_ch == "EEGsec", 1);  % C3-A2
    idx_C4    = find(labels_ch == "EEG",    1);  % C4-A1
    idx_EOG_L = find(labels_ch == "EOGL",   1);  % Left EOG
    idx_EOG_R = find(labels_ch == "EOGR",   1);  % Right EOG
    idx_EMG   = find(labels_ch == "EMG",    1);  % Chin EMG

    % === EEG (C3 + C4 combined, like training) ===
    if isempty(idx_C3) && isempty(idx_C4)
        warning('Record %s: no EEGsec/EEG channels found; skipping.', recName);
        continue;
    end

    eeg_idx = [idx_C3, idx_C4];
    eeg_idx = eeg_idx(~isnan(eeg_idx));  % keep existing ones

    if ~isempty(idx_C3) && ~isempty(idx_C4)
        eeg_signal_comb = (record(idx_C3,:) + record(idx_C4,:)) / 2;
        fprintf(' EEG: using AVERAGE of %s and %s\n', hdr.label{idx_C3}, hdr.label{idx_C4});
    else
        idx_single = eeg_idx(1);
        eeg_signal_comb = record(idx_single,:);
        fprintf(' EEG: using single channel %s\n', hdr.label{idx_single});
    end

    fs_eeg = double(hdr.samples(eeg_idx(1))) / double(hdr.duration);
    samplesPerEpochEEG = round(fs_eeg * epochLenSec);

    % --- Number of EEG epochs from signal length ---
    nSamplesEEG = numel(eeg_signal_comb);
    nEpochs_eeg = floor(nSamplesEEG / samplesPerEpochEEG);
    fprintf(' EEG signal length = %d samples -> %d epochs\n', nSamplesEEG, nEpochs_eeg);

    if nEpochs_eeg == 0
        warning('Record %s: no complete EEG epochs; skipping.', recName);
        continue;
    end

    % === EOG ===
    have_eog = ~isempty(idx_EOG_L) && ~isempty(idx_EOG_R);
    if have_eog
        fs_eog = double(hdr.samples(idx_EOG_L)) / double(hdr.duration);
        samplesPerEpochEOG = round(fs_eog * epochLenSec);
        fprintf(' EOG: using %s and %s\n', hdr.label{idx_EOG_L}, hdr.label{idx_EOG_R});
    end

    % === EMG ===
    have_emg = ~isempty(idx_EMG);
    if have_emg
        fs_emg = double(hdr.samples(idx_EMG)) / double(hdr.duration);
        samplesPerEpochEMG = round(fs_emg * epochLenSec);
        fprintf(' EMG: using %s\n', hdr.label{idx_EMG});
    end

    % Decide common number of epochs based on all channels that exist
    nEpochs = nEpochs_eeg;

    if have_eog
        sig_eogL = record(idx_EOG_L,:);
        nEpochs_eogL = floor(numel(sig_eogL) / samplesPerEpochEOG);
        sig_eogR = record(idx_EOG_R,:);
        nEpochs_eogR = floor(numel(sig_eogR) / samplesPerEpochEOG);
        nEpochs = min([nEpochs, nEpochs_eogL, nEpochs_eogR]);
    end

    if have_emg
        sig_emg = record(idx_EMG,:);
        nEpochs_emg = floor(numel(sig_emg) / samplesPerEpochEMG);
        nEpochs = min([nEpochs, nEpochs_emg]);
    end

    if nEpochs == 0
        warning('Record %s: no common complete epochs across channels; skipping.', recName);
        continue;
    end

    fprintf(' Using %d complete epochs for ALL channels.\n', nEpochs);

    %% --- Build EEG epochs (multi-channel, but no labels/valid mask) ---
    nEEG = numel(eeg_idx);
    eeg_epochs_mc = zeros(nEpochs, nEEG, samplesPerEpochEEG);

    for ch = 1:nEEG
        sig = record(eeg_idx(ch), :);
        for e = 1:nEpochs
            s_start = (e-1)*samplesPerEpochEEG + 1;
            s_end   = e*samplesPerEpochEEG;
            eeg_epochs_mc(e, ch, :) = sig(s_start:s_end);
        end
    end

    %% --- Build EOG epochs ---
    if have_eog
        nEOG = 2;
        eog_epochs_mc = zeros(nEpochs, nEOG, samplesPerEpochEOG);

        sig_L = record(idx_EOG_L, :);
        sig_R = record(idx_EOG_R, :);

        for e = 1:nEpochs
            s_start = (e-1)*samplesPerEpochEOG + 1;
            s_end   = e*samplesPerEpochEOG;
            eog_epochs_mc(e, 1, :) = sig_L(s_start:s_end);
            eog_epochs_mc(e, 2, :) = sig_R(s_start:s_end);
        end
    else
        eog_epochs_mc = [];
    end

    %% --- Build EMG epochs ---
    if have_emg
        nEMG = 1;
        emg_epochs_mc = zeros(nEpochs, nEMG, samplesPerEpochEMG);

        sig = record(idx_EMG, :);
        for e = 1:nEpochs
            s_start = (e-1)*samplesPerEpochEMG + 1;
            s_end   = e*samplesPerEpochEMG;
            emg_epochs_mc(e, 1, :) = sig(s_start:s_end);
        end
    else
        emg_epochs_mc = [];
    end

    %% --- Accumulate across records ---
    eeg_mc_all{end+1} = eeg_epochs_mc; %#ok<AGROW>
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

    % Record <-> epoch tracking for submission
    record_numbers_all{end+1} = repmat(string(recName), nEpochs, 1); %#ok<AGROW>
    epoch_numbers_all{end+1}  = (1:nEpochs)';                          %#ok<AGROW>
    subject_ids_all{end+1}    = r * ones(nEpochs, 1);                  %#ok<AGROW>
end

%% --- Concatenate across all holdout records ---
multi_channel_data = struct();
if any(~cellfun(@isempty, eeg_mc_all))
    multi_channel_data.eeg = cat(1, eeg_mc_all{:});
end
if any(~cellfun(@isempty, eog_mc_all))
    % Some records might have empty EOG, keep only non-empty
    nonEmptyEog = ~cellfun(@isempty, eog_mc_all);
    if any(nonEmptyEog)
        multi_channel_data.eog = cat(1, eog_mc_all{nonEmptyEog});
    end
end
if any(~cellfun(@isempty, emg_mc_all))
    nonEmptyEmg = ~cellfun(@isempty, emg_mc_all);
    if any(nonEmptyEmg)
        multi_channel_data.emg = cat(1, emg_mc_all{nonEmptyEmg});
    end
end

record_numbers = cat(1, record_numbers_all{:});
epoch_numbers  = cat(1, epoch_numbers_all{:});
subject_ids    = cat(1, subject_ids_all{:});

fprintf('\nFinished building HOLDOUT data.\n');
if isfield(multi_channel_data, 'eeg')
    fprintf(' Total epochs: %d (EEG)\n', size(multi_channel_data.eeg, 1));
end

%% --- channel_info for holdout (similar to training) ---
channel_info = struct();
if isfield(multi_channel_data,'eeg')
    channel_info.eeg.names = {};
    channel_info.eeg.fs    = fs_eeg;
end
if ~isempty(fs_eog) && isfield(multi_channel_data,'eog')
    channel_info.eog.names = {};
    channel_info.eog.fs    = fs_eog;
end
if ~isempty(fs_emg) && isfield(multi_channel_data,'emg')
    channel_info.emg.names = {};
    channel_info.emg.fs    = fs_emg;
end

channel_info.subject_ids   = subject_ids;
channel_info.subject_names = subject_names;

end
