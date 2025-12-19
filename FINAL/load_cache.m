function data = load_cache(filename, cache_dir)
%LOAD_CACHE Load a cached variable if it exists, otherwise return [].
%
%   data = load_cache(filename, cache_dir)
%
%   filename  - name of the .mat file, e.g. 'features_iter1.mat'
%   cache_dir - folder where the cache file is stored

    if nargin < 2 || isempty(cache_dir)
        cache_dir = '.';
    end

    fullpath = fullfile(cache_dir, filename);

    if exist(fullpath, 'file')
        % Load everything and be flexible about variable name
        s = load(fullpath);   % load all variables

        if isfield(s, 'data')
            data = s.data;
        elseif isfield(s, 'model')
            % Old cache files may have been saved as 'model'
            data = s.model;
        else
            warning('Cache file %s has no ''data'' or ''model'' variable. Ignoring it.', fullpath);
            data = [];
        end

        fprintf('Loaded cache from %s\n', fullpath);
    else
        data = [];  % so "isempty(data)" is true in main.m
        fprintf('No cache file found at %s\n', fullpath);
    end
end
