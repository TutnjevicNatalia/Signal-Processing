function plot_hypnogram_comparison(y_true, y_pred, epoch_length_sec, subj_name)
% PLOT_HYPNOGRAM_COMPARISON
% Visualises ground truth vs predicted sleep stages for one subject.
%
% INPUTS:
%   y_true  - true sleep stage labels (Nx1 or Nx1)
%   y_pred  - predicted sleep stage labels (Nx1)
%   epoch_length_sec - epoch duration, usually 30 sec
%   subj_name - string for plot title, e.g. 'R3'
%
% OUTPUT:
%   A figure with two subplots:
%       (1) Ground truth hypnogram
%       (2) Predicted hypnogram
%
% NOTE:
%   Sleep stages must already be integers encoded as:
%   0 = Wake, 1 = N1, 2 = N2, 3 = N3, 4 = REM

    if nargin < 3
        epoch_length_sec = 30; % default
    end
    if nargin < 4
        subj_name = '';
    end

    % Convert to column vectors
    y_true = y_true(:);
    y_pred = y_pred(:);

    % Time axis in hours
    N = length(y_true);
    time_hours = (0:N-1) * (epoch_length_sec / 3600.0);

    % Stage labels
    stage_names = {'W', 'N1', 'N2', 'N3', 'REM'};

    % Plot settings
    figure('Name', ['Hypnogram: ' subj_name], 'Color', 'w', 'Position', [200 200 1200 350]);

    % === 1. True Hypnogram ===
    subplot(2,1,1);
    stairs(time_hours, y_true, 'LineWidth', 1.5);
    ylim([-0.5 4.5]);
    yticks(0:4);
    yticklabels(stage_names);
    xlabel('Time (hours)');
    ylabel('Stage');
    title(['True Hypnogram - ' subj_name]);
    grid on;

    % === 2. Predicted Hypnogram ===
    subplot(2,1,2);
    stairs(time_hours, y_pred, 'LineWidth', 1.5);
    ylim([-0.5 4.5]);
    yticks(0:4);
    yticklabels(stage_names);
    xlabel('Time (hours)');
    ylabel('Stage');
    title(['Predicted Hypnogram - ' subj_name]);
    grid on;
end
