function y_smooth = smooth_predictions_median(y, win)
% SMOOTH_PREDICTIONS_MEDIAN
% Median-filter temporal smoothing for sleep stage predictions.
%
% Inputs:
%   y   : [N x 1] predicted labels (numeric, e.g. 0–4 or 1–5)
%   win : odd integer window length (e.g. 3 or 5)
%
% Output:
%   y_smooth : smoothed labels, same size as y
%
% Notes:
%   - Smoothing is applied ONLY in time
%   - Should be used per subject/record (NOT across recordings)
%   - Median filter removes isolated "salt-and-pepper" errors

    if nargin < 2
        win = 5;
    end

    if mod(win,2) == 0
        error('Window size win must be odd.');
    end

    y = y(:);  % ensure column
    N = numel(y);

    y_smooth = y;

    half = floor(win/2);

    for i = 1:N
        i1 = max(1, i-half);
        i2 = min(N, i+half);
        y_smooth(i) = median(y(i1:i2));
    end
end
