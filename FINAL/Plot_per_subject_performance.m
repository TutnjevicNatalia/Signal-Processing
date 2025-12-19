%% plot_holdout_metrics_all_iterations.m
% Creates 2 figures (Accuracy, Macro F1), each with 4 subplots:
% Iteration 1-4 (H1-H10 bars)

clc; close all; clear;

subjects = {'H1','H2','H3','H4','H5','H6','H7','H8','H9','H10'};
iters    = {'Iteration 1','Iteration 2','Iteration 3','Iteration 4'};

%% ------------------ DATA ------------------
% Accuracy (%): [10 records x 4 iterations]
acc = [
    38.02  33.10  31.34  30.45;  % H1
    36.52  37.70  33.56  33.86;  % H2
    38.46  49.06  47.00  49.06;  % H3
    27.72  36.52  32.68  34.27;  % H4
    35.76  40.37  43.03  45.68;  % H5
    45.30  46.87  52.30  53.76;  % H6
    31.49  33.08  36.46  35.05;  % H7
    32.83  32.36  35.08  33.77;  % H8
    41.29  46.16  44.76  44.01;  % H9
    32.74  38.46  37.71  39.49   % H10
];

% Macro F1 (%): [10 records x 4 iterations]
f1_percent = [
    23.35  22.32  22.83  20.64;  % H1
    24.07  24.45  24.54  23.84;  % H2
    24.79  31.06  32.35  37.10;  % H3
    21.16  27.20  26.68  26.25;  % H4
    25.56  30.44  33.96  35.22;  % H5
    28.09  30.62  31.32  31.26;  % H6
    21.54  23.69  25.52  25.70;  % H7
    20.96  17.52  19.58  17.61;  % H8
    19.55  21.65  21.35  20.66;  % H9
    21.12  24.56  21.66  22.37   % H10
];

%% ------------------ PLOTTING HELPER ------------------
plot4 = @(data, ylab, figName, ylims) plot_four_subplots(subjects, iters, data, ylab, figName, ylims);

%% Figure 1: Accuracy
plot4(acc, 'Accuracy (%)', 'Holdout Accuracy (Iterations 1-4)', [0 60]);

%% Figure 2: Macro F1 (%)
plot4(f1_percent, 'Macro F1 (%)', 'Holdout Macro F1 (Iterations 1-4)', [0 40]);

%% ------------------ LOCAL FUNCTION ------------------
function plot_four_subplots(subjects, iters, data, ylab, figTitle, ylims)
    figure('Color','w','Name',figTitle);

    for i = 1:4
        subplot(2,2,i);
        bar(data(:,i));
        grid on;

        set(gca, 'XTick', 1:numel(subjects), ...
                 'XTickLabel', subjects, ...
                 'FontSize', 10);

        ylabel(ylab);
        title(iters{i});
        if ~isempty(ylims)
            ylim(ylims);
        end
    end

    sgtitle(figTitle);
end
