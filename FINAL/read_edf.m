function [hdr_all, record_all] = read_edf(filePath)
    % READ_EDF  Läser alla R1..R10.edf i katalogen filePath
    %
    % Output:
    %   hdr_all    : cell-array {1x10}, varje cell är ett hdr-struct från edfread
    %   record_all : cell-array {1x10}, varje cell är [nCh x nSamples] double

   recNames = {'R1', 'R2', 'R3', 'R4', 'R5', 'R6', 'R7', 'R8', 'R9', 'R10'};

    nRecs = numel(recNames);

    hdr_all    = cell(1, nRecs);
    record_all = cell(1, nRecs);

    ampThresh = 5e3;   % amplitude threshold för "obviously bad" samples

    for r = 1:nRecs
        recName = recNames{r};
        fprintf('\n==============================\n');
        fprintf('Processing %s\n', recName);
        fprintf('==============================\n');

        %% read edf 
        edf_fileName = [recName '.edf'];
        edf_filePath = fullfile(filePath, edf_fileName)

        [hdr, record] = edfread(edf_filePath);
        record = double(record);   % jobba i double

        % spara hdr och signal
        hdr_all{r}    = hdr;
        record_all{r} = record;

        %% ---------- PRINT CHANNEL INFO ----------
        fs = hdr.samples(:) ./ hdr.duration; 
        duration = hdr.duration; 

        for i = 1:numel(hdr.label)
            fprintf('Channel %2d: %-12s | fs = %.2f Hz\n', i, hdr.label{i}, fs(i));
        end

        labels = string(hdr.label);
        for i = 1:numel(labels)
            fprintf('Ch %2d: %-14s | samples/rec = %4d | rec dur = %.2f s | fs = %.2f Hz\n', ...
                i, labels(i), hdr.samples(i), hdr.duration, fs(i));
        end

        %% Checking for missing data 
        [nCh, ~] = size(record);

        nanCount     = sum(isnan(record), 2);
        infCount     = sum(isinf(record),  2);
        outlierCount = sum(abs(record) > ampThresh, 2);

        fprintf('\n=== Data quality BEFORE cleaning (%s) ===\n', recName);
        for ch = 1:nCh
            fprintf('Ch %2d (%-12s): NaN=%5d, Inf=%5d, |x|>%g =%5d\n', ...
                ch, hdr.label{ch}, nanCount(ch), infCount(ch), ampThresh, outlierCount(ch));
        end
    end  

    fprintf('\nDone reading all EDFs in %s\n', filePath);
end