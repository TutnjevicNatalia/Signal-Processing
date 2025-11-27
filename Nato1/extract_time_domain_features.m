function features = extract_time_domain_features(epoch)
%% Complete time-domain feature set (16 features)

f_mean = mean(epoch);
f_median = median(epoch);
f_std = std(epoch);
f_var = var(epoch);
f_rms = sqrt(mean(epoch.^2));
f_min = min(epoch);
f_max = max(epoch);
f_range = f_max - f_min;
f_skew = skewness(epoch);
f_kurt = kurtosis(epoch);

% Zero-crossings
f_zc = sum(abs(diff(epoch > 0)));

% Hjorth parameters
activity = var(epoch);
mobility = sqrt(var(diff(epoch)) / activity);
complexity = sqrt(var(diff(diff(epoch))) / var(diff(epoch))) / mobility;

% Energy / power
f_energy = sum(epoch.^2);
f_power = mean(epoch.^2);

features = [
    f_mean, f_median, f_std, f_var, f_rms, ...
    f_min, f_max, f_range, f_skew, f_kurt, ...
    f_zc, activity, mobility, complexity, ...
    f_energy, f_power
];
end