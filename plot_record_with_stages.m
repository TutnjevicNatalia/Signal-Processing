function plot_record_with_stages(hdr, record, stages, epochLength)
% PLOT_RECORD_WITH_STAGES Plot all channels and the sleep stages.
%
%   hdr         : header struct from edfread
%   signals     : [nCh x nSamples] matrix (cleaned signals)
%   stages      : vector of sleep stages (from readXML, 0..5)
%   epochLength : epoch length in seconds from XML (not used for stages here,
%                 because readXML builds stages at 1 sample per SECOND)
%
% Example use:
%   plot_record_with_stages(hdr_all{1}, signals_all{1}, stages_all{1}, epochLength_all{1});

    % ---------- BASIC INFO ----------
    [nCh, nSamples] = size(record);

    % Use fs from the first channel (common in sleep EDFs)
    fs1 = double(hdr.samples(1)) / double(hdr.duration);  % Hz

    % Time axis for signals in hours
    t_sig = (0:nSamples-1) / fs1 / 3600;  % hours

    % Time axis for stages.
    % readXML builds one stage value per SECOND (duration in seconds),
    % so treat each entry in "stages" as 1 second:
    nStageSamples = numel(stages);
    t_stage = (0:nStageSamples-1) / 3600;   % seconds -> hours

    % ---------- PLOT SIGNALS ----------
    figure(1); clf;

    % Number of rows = all channels
    nRows = nCh;

    for ch = 1:nCh
        subplot(nRows, 1, ch);

        plot(t_sig, record(ch,:));
        xlim([t_sig(1) t_sig(end)]);
        ylabel(strtrim(hdr.label{ch}), 'Interpreter','none');

        if ch < nCh
            set(gca, 'XTickLabel', []);  % hide x labels except last
        else
            xlabel('Time (hours)');
        end

      
    end

    % ---------- PLOT SLEEP STAGES ----------
    figure(2); clf;

    % Use stairs so you see the step-like hypnogram
    stairs(t_stage, stages, 'LineWidth', 1.5);

    % Match x-limits roughly to the signal duration
    xlim([t_sig(1) t_sig(end)]);

    % Our coding from readXML:
    % 0 = REM, 1 = N4, 2 = N3, 3 = N2, 4 = N1, 5 = Wake
    yticks(0:5);
    yticklabels({'REM','N4','N3','N2','N1','Wake'});

    xlabel('Time (hours)');
    ylabel('Stage');

    % Put Wake at the top visually
    set(gca, 'YDir','reverse');

    title('Sleep Stages');
end
