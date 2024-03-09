function [hotPixelCoords, noisePixelCoords] = identifyMostLinearHotPixel(windowContents, windowCoords, firingRateThreshold)
    correlationThreshold = 0.9;
    R2Threshold = 0.99;
    % firingRateThreshold = 15; % Minimum number of timestamps for a hot pixel
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
    [hasLowCorrelation, allStronglyCorrelated] = checkLowCorrelation(flatTimeline, correlationThreshold, numBins);
    
    % If all pixels are firing at the same rate, then do not consider as a
    % hot pixel or noise, it must be an active source of information
    % triggering too many events
    if allStronglyCorrelated
        hotPixelCoords = {};
        noisePixelCoords = {};
        return; % Early exit if there's high uniform correlation among pixels
    end

    % Process each pixel only if overall correlation is low
    if hasLowCorrelation
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

function [hasLowCorrelation, allStronglyCorrelated] = checkLowCorrelation(flatContents, correlationThreshold, numBins)
    minTime = inf;
    maxTime = -inf;
    for content = flatContents
        if ~isempty(content{1})
            minTime = min(minTime, min(content{1}));
            maxTime = max(maxTime, max(content{1}));
        end
    end
    
    if minTime == maxTime
        hasLowCorrelation = false;
        allStronglyCorrelated = true; % No variability implies strong correlation.
        return;
    end
    
    binnedData = zeros(numel(flatContents), numBins);
    for i = 1:numel(flatContents)
        if ~isempty(flatContents{i})
            normalizedTimestamps = (flatContents{i} - minTime) / (maxTime - minTime);
            binIndices = ceil(normalizedTimestamps * numBins);
            binIndices(binIndices < 1) = 1;
            binIndices(binIndices > numBins) = numBins;
            for index = binIndices
                binnedData(i, index) = binnedData(i, index) + 1;
            end
        end
    end
    
    corrMatrix = corr(binnedData');
    avgCorrelation = mean(corrMatrix(~isnan(corrMatrix) & ~isinf(corrMatrix) & corrMatrix~=1));
    
    % Determine conditions
    hasLowCorrelation = any(avgCorrelation < correlationThreshold);
    allStronglyCorrelated = all(avgCorrelation >= correlationThreshold) && ~isempty(avgCorrelation);
end

