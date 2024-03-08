function [hotPixelCoords, noisePixelCoords] = identifyMostLinearHotPixel(windowContents, windowCoords)
    correlationThreshold = 0.99;
    R2Threshold = 0.99;
    firingRateThreshold = 10; % Minimum number of timestamps for a hot pixel
    hotPixelData = {}; % Initialize to store hot pixels data (coords and R2)
    noisePixelData = {}; % Initialize to store noise pixels data (coords)
    
    % Flatten windowContents for easier manipulation, including empty cells
    flatTimeline = reshape(windowContents, 1, numel(windowContents));
    
    % Track the number of non-empty timelines to identify single firing pixels
    nonEmptyTimelines = sum(~cellfun(@isempty, flatTimeline));
    
    % If there is exactly one firing pixel, directly classify based on its timestamps
    if nonEmptyTimelines == 1
        idx = find(~cellfun(@isempty, flatTimeline), 1);
        timestamps = flatTimeline{idx};
        if numel(timestamps) > firingRateThreshold
            % Run timestamps through linear regression
            X = (1:numel(timestamps))'; % Independent variable (time order)
            Y = timestamps'; % Dependent variable (timestamps)

            % Fit linear regression and calculate R^2
            mdl = fitlm(X, Y);
            R2 = mdl.Rsquared.Ordinary;

            % Classify based on R^2 threshold
            if R2 >= R2Threshold
                hotPixelData{end+1} = struct('Coords', windowCoords{idx}, 'R2', R2);
            else
                noisePixelData{end+1} = struct('Coords', windowCoords{idx});
            end
        else
            % If the pixel fires below the firing rate threshold, classify as noise
            noisePixelData{end+1} = struct('Coords', windowCoords{idx});
        end
        % Early return since the single pixel scenario is handled
        hotPixelCoords = extractCoords(hotPixelData);
        noisePixelCoords = extractCoords(noisePixelData);
        return;
    end


    % Presuming numBins, correlationThreshold, R2Threshold, and firingRateThreshold are defined above
    numBins = 100; % Number of bins for binning timestamps
    isLowCorrelation = checkLowCorrelation(flatTimeline, correlationThreshold, numBins);

    % Process each pixel only if overall correlation is low
    if isLowCorrelation
        for i = 1:numel(flatTimeline)
            timestamps = flatTimeline{i};
            
            % Proceed only if the pixel has more than one timestamp
            if ~isempty(timestamps) && numel(timestamps) > 1
                % Check if the pixel's firing rate exceeds the firing rate threshold
                if numel(timestamps) > firingRateThreshold
                    % Prepare data for linear regression
                    X = (1:numel(timestamps))'; % Independent variable (time order)
                    Y = timestamps'; % Dependent variable (timestamps)

                    % Fit linear regression and calculate R^2
                    mdl = fitlm(X, Y);
                    R2 = mdl.Rsquared.Ordinary;

                    % Classify based on R^2 value
                    if R2 >= R2Threshold
                        hotPixelData{end+1} = struct('Coords', windowCoords{i}, 'R2', R2);
                    % else
                    %     noisePixelData{end+1} = struct('Coords', windowCoords{i});
                    end
                end
                % Skip the pixel if its firing rate is below or equal to the threshold
                % This condition is implicitly handled by not adding it to either hotPixelData or noisePixelData
            else
                noisePixelData{end+1} = struct('Coords', windowCoords{i});
            end
        end
    end
    % Extract and return coordinates of hot pixels and noise pixels
    hotPixelCoords = extractCoords(hotPixelData);
    noisePixelCoords = extractCoords(noisePixelData);
end

function coords = extractCoords(pixelData)
    coords = {};
    for i = 1:length(pixelData)
        if isfield(pixelData{i}, 'Coords')
            coords{end+1} = pixelData{i}.Coords;
        end
    end
end

function isLowCorrelation = checkLowCorrelation(flatContents, correlationThreshold, numBins)
    % Define the common time range and bin the timestamps
    minTime = inf;
    maxTime = -inf;
    for content = flatContents
        if ~isempty(content{1})
            minTime = min(minTime, min(content{1}));
            maxTime = max(maxTime, max(content{1}));
        end
    end

    % Avoid division by zero in case all timestamps are identical
    if minTime == maxTime
        isLowCorrelation = true; % Consider as low correlation if there's not enough variability
        return;
    end

    % Bin the timestamps into a common set of intervals
    binnedData = zeros(numel(flatContents), numBins);
    for i = 1:numel(flatContents)
        if ~isempty(flatContents{i})
            normalizedTimestamps = (flatContents{i} - minTime) / (maxTime - minTime);
            binIndices = ceil(normalizedTimestamps * numBins);
            binIndices(binIndices < 1) = 1; % Ensure indices are within bounds
            binIndices(binIndices > numBins) = numBins;
            for index = binIndices
                binnedData(i, index) = binnedData(i, index) + 1;
            end
        end
    end

    % Calculate the pairwise correlation matrix from the binned data
    corrMatrix = corr(binnedData');

    % Calculate average correlation excluding NaN values and self-correlations
    corrMatrix(tril(true(size(corrMatrix)))) = NaN; % Ignore lower triangle and diagonal
    avgCorrelation = nanmean(corrMatrix(:));

    % Determine if the overall correlation is considered low
    isLowCorrelation = avgCorrelation < correlationThreshold;
end
