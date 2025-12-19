function save_cache(data, filename, cache_dir)
%SAVE_CACHE Save a variable to a .mat cache file as 'data'.
%
%   save_cache(data, filename, cache_dir)

    if nargin < 3 || isempty(cache_dir)
        cache_dir = '.';
    end

    if ~exist(cache_dir, 'dir')
        mkdir(cache_dir);
    end

    fullpath = fullfile(cache_dir, filename);

    % Always save under the name 'data'
    save(fullpath, 'data', '-v7.3');

    fprintf('Saved cache to %s\n', fullpath);
end
