% load("focus_td_dataForDesign.mat");td.ts = td.ts - td.ts(1);
% load('/home/samiarja/Desktop/PhD/Dataset/NORALPH_ICNS_EB_Space_Imaging_Speed_Dataset/2022-03-31T11-51-20Z_speed_survey-WINDY/files/2022-03-31T11-52-19Z_speed_survey_191.930378_-59.688764_0.125/psee400.mat')
% load('/home/samiarja/Desktop/PhD/Dataset/NORALPH_ICNS_EB_Space_Imaging_Speed_Dataset/2022-03-31T11-51-20Z_speed_survey-WINDY/files/2022-03-31T11-54-28Z_speed_survey_191.930378_-59.688764_0.00390625/psee400.mat')
% load('/home/samiarja/Desktop/PhD/Dataset/NORALPH_ICNS_EB_Space_Imaging_Speed_Dataset/2022-03-29T11-43-31Z_speed_survey-REGULAR/files/2022-03-29T11-52-30Z_speed_survey_114.825500_5.224993_0.0009765625/psee400.mat')
% load('/home/samiarja/Desktop/PhD/Dataset/NORALPH_ICNS_EB_Space_Imaging_Speed_Dataset/2022-03-31T11-51-20Z_speed_survey-WINDY/files/2022-03-31T11-58-22Z_speed_survey_191.930378_-59.688764_7.62939453125e-06/psee400.mat')
% load('/home/samiarja/Desktop/PhD/Dataset/NORALPH_ICNS_EB_Space_Imaging_Speed_Dataset/2022-03-31T11-51-20Z_speed_survey-WINDY/files/2022-03-31T11-56-38Z_speed_survey_191.930378_-59.688764_0.0001220703125/psee400.mat')
% load('/home/samiarja/Desktop/PhD/Dataset/NORALPH_ICNS_EB_Space_Imaging_Speed_Dataset/2022-03-31T11-51-20Z_speed_survey-WINDY/files/2022-03-31T11-55-46Z_speed_survey_191.930378_-59.688764_0.00048828125/psee400.mat')
load('/home/samiarja/Desktop/PhD/Dataset/NORALPH_ICNS_EB_Space_Imaging_Speed_Dataset/2022-03-31T11-51-20Z_speed_survey-WINDY/files/2022-03-31T11-53-06Z_speed_survey_191.930378_-59.688764_0.03125/psee400.mat')

td = struct("x",double(events(:,2)),"y",double(events(:,3)),"p",double(events(:,4)),"ts",double(events(:,1)));
td.x = td.x+1;td.y = td.y+1;


fig = 0;
RoiSize = 3;
firingRateThreshold = 10;
nEvent = numel(td.x);
tListCell = cell(RoiSize,RoiSize);
xMax = max(td.x);yMax = max(td.y);
tCell = cell(xMax,yMax);
eventCount = zeros(xMax,yMax);
td.h = td.x+nan; % 1=hotpixels, 2=noise

pixel_map = cell(RoiSize, RoiSize);

for ii = 1:nEvent % this need to be event chunks
    x = td.x(ii);
    y = td.y(ii);
    t = td.ts(ii);
    eventCount(x,y) = eventCount(x,y)+1;
    tCell{x,y}(eventCount(x,y)) = t;
end

startTime = tic;

noisy_events = nan(1e4,2);
hotpix_events = nan(1e4,2);
noise_counter = 0;
hot_counter = 0;

number_of_hot_pixels = 0;
number_of_noise_pixels = 0;
totalIterations = ((xMax-(RoiSize-1))/RoiSize) * ((yMax-(RoiSize-1))/RoiSize);
iterationsCompleted = 0;
for x = 1:RoiSize:xMax-(RoiSize-1)
    for y = 1:RoiSize:yMax-(RoiSize-1)
        iterationsCompleted = iterationsCompleted + 1;
        windowContents = cell(RoiSize, RoiSize); % To store the arrays within the window
        windowCoords = cell(RoiSize, RoiSize); % To store the coordinates of the arrays within the window
        hasActiveEnough = false; % Flag to indicate if any cell in the window has more than 3 values

        % Accumulate the contents of the 3x3 window
        for i = 0:RoiSize-1
            for j = 0:RoiSize-1
                currentCell = tCell{x+i, y+j};
                windowContents{i+1, j+1} = currentCell; % Store the array
                windowCoords{i+1, j+1} = [x+i, y+j];    % Store the coordinates
            end
        end
        
        % Check if at least one cell in the window firing enough
        for i = 1:RoiSize
            for j = 1:RoiSize
                if numel(windowContents{i, j}) > firingRateThreshold
                    hasActiveEnough = true;
                    break; % Exit the inner loop if condition is met
                end
            end
            if hasActiveEnough
                break; % Exit the outer loop if condition is met
            end
        end
        
        % If the condition is met, you can perform your desired action here
        if hasActiveEnough
            % Example action: Print the top-left coordinate of the window
            % disp(['Window with a cell having more than 3 elements starts at: (', num2str(x), ', ', num2str(y), ')']);
            
            extractedSection = zeros(RoiSize, RoiSize);

            for i = 1:RoiSize
                for j = 1:RoiSize
                    coord = windowCoords{i, j};
                    xcoor = coord(1);
                    ycoor = coord(2);
                    extractedSection(i, j) = eventCount(xcoor, ycoor);
                end
            end

            % Do whatever you want here
            % isolate hot pixels
            % hotPixelCoords = identifyHotPixels(windowContents, windowCoords);
            [hotPixelCoords, noisePixelCoords] = identifyMostLinearHotPixel(windowContents, windowCoords, firingRateThreshold);
            % hotPixelCoords = identifyHotPixelsWithChangeDetection(windowContents, windowCoords);

            if ~isempty(hotPixelCoords)
                number_of_hot_pixels = number_of_hot_pixels+numel(hotPixelCoords);
            end
            % disp(['Hot Pixels Detected: ', num2str(number_of_hot_pixels)])
            
            if ~isempty(noisePixelCoords)
                number_of_noise_pixels = number_of_noise_pixels+numel(noisePixelCoords);
            end
            % disp(['Noisy Pixels Detected: ', num2str(number_of_noise_pixels)])

            if fig
                figure(56767); clf;
                subplot(1,3,1); hold on; grid on; title('Window Contents');
                for i = 1:numel(windowContents)
                    if ~isempty(windowContents{i})
                        plot(windowContents{i});hold on
                    end
                end
            end
            
            if fig
                subplot(1,3,2);
                imagesc(log(extractedSection)); % Using log to enhance contrast, adding 1 to avoid log(0)
                colorbar; title('Extracted Section');colormap default
                axis image;
            end
            
            
            range = (td.x>windowCoords{1,1}(1)-1 & ...
                     td.x<windowCoords{3,1}(1)+1 & ...
                     td.y>windowCoords{1,1}(2)-1 & ...
                     td.y<windowCoords{3,3}(2)+1);
            
            if ~isempty(hotPixelCoords)
                for hp = 1:numel(hotPixelCoords)
                    hp_coor = hotPixelCoords{hp};
                    hot_counter=hot_counter+1;
                    hotpix_events(hot_counter,:) = hp_coor;
                    hotpixbound = (td.x==hp_coor(1) & td.y==hp_coor(2));

                    onlyhot = (range & hotpixbound);
                    td.h(onlyhot) = 1;
                    
                    if fig
                        subplot(1,3,3)
                        plot3(td.x(range),td.y(range),td.ts(range),".b","MarkerSize",5);grid on;
                        hold on;plot3(td.x(onlyhot),td.y(onlyhot),td.ts(onlyhot),".r","MarkerSize",5);
                    end
                end
            elseif ~isempty(noisePixelCoords)
                for hp = 1:numel(noisePixelCoords)
                    noise_coor = noisePixelCoords{hp};
                    noise_counter=noise_counter+1;
                    noisy_events(noise_counter,:) = noise_coor;
                    noisepixbound = (td.x==noise_coor(1) & td.y==noise_coor(2));

                    onlynoise = (range & noisepixbound);
                    td.h(onlynoise) = 2;
                    
                    if fig
                        plot3(td.x(range),td.y(range),td.ts(range),".b","MarkerSize",5);grid on;
                        hold on;plot3(td.x(onlynoise),td.y(onlynoise),td.ts(onlynoise),".g","MarkerSize",5);
                    end
                end
            else
                if fig
                    plot3(td.x(range),td.y(range),td.ts(range),".b","MarkerSize",5);grid on;
                end
            end
            drawnow;
            
            if fig
                figure(45667);subplot(1,2,1)
                imagesc(log(eventCount));hold on;axis image
                rectangle('Position',[windowCoords{2,2}(2)-RoiSize/2 windowCoords{2,2}(1)-RoiSize/2 RoiSize RoiSize],'EdgeColor','r')
                subplot(1,2,2);imagesc(log(eventCount));hold on
                rectangle('Position',[windowCoords{2,2}(2)-RoiSize/2 windowCoords{2,2}(1)-RoiSize/2 RoiSize RoiSize],'EdgeColor','r')
                xLimLower = max(1, windowCoords{2,2}(1) - RoiSize*10);
                xLimUpper = min(size(eventCount, 2), windowCoords{2,2}(1) + RoiSize*10);
                yLimLower = max(1, windowCoords{2,2}(2) - RoiSize*10);
                yLimUpper = min(size(eventCount, 1), windowCoords{2,2}(2) + RoiSize*10);
                ylim([xLimLower, xLimUpper]);
                xlim([yLimLower, yLimUpper]);
                drawnow;
            end
        end

        if mod(iterationsCompleted, 10) == 0 % Update progress every n iterations
            elapsedTime = toc(startTime);
            estimatedTotalTime = (elapsedTime / iterationsCompleted) * totalIterations;
            remainingTime = estimatedTotalTime - elapsedTime;
            progressPercentage = (iterationsCompleted / totalIterations) * 100;
            
            % Display progress
            fprintf('Progress: %.2f%%. Estimated remaining time: %.2f seconds.\n', ...
                    progressPercentage, remainingTime);
        end
    end
end

number_hot_pix   = ~isnan(hotpix_events(:,1));
number_noise_pix = ~isnan(noisy_events(:,1));

disp("Total hot pixels: " + num2str(numel(number_hot_pix(number_hot_pix>0))))
disp("Total noise pixels: " + num2str(numel(number_noise_pix(number_noise_pix>0))))

eventCount_raw = zeros(yMax,xMax);
for ii = 1:numel(td.x)
    x = td.x(ii);
    y = td.y(ii);
    t = td.ts(ii);
    eventCount_raw(y,x) = eventCount_raw(y,x)+1;
end
eventCount_cleaned = zeros(yMax,xMax);
for ii = 1:numel(td.x)
    x = td.x(ii);
    y = td.y(ii);
    t = td.ts(ii);
    h = td.h(ii);
    if isnan(h)
        eventCount_cleaned(y,x) = eventCount_cleaned(y,x)+1;
    end
end

figure(56756);
ax(1)=subplot(2,3,1);
plot3(td.x,td.y,td.ts,".b","MarkerSize",1);grid on;
title("Original events");xlabel("x");ylabel("y");zlabel("t")
ylim([0 max(td.y)]);

ax(2)=subplot(2,3,2);
plot3(td.x(isnan(td.h)),td.y(isnan(td.h)),td.ts(isnan(td.h)),".b","MarkerSize",1);grid on;hold on
plot3(td.x(td.h==1),td.y(td.h==1),td.ts(td.h==1),".r","MarkerSize",5);
plot3(td.x(td.h==2),td.y(td.h==2),td.ts(td.h==2),".g","MarkerSize",5);
legend("Signal", "Hotpix", "Noise"); title("Labeled events");
xlabel("x");ylabel("y");zlabel("t")
ylim([0 max(td.y)]);

ax(3)=subplot(2,3,3);
plot3(td.x(isnan(td.h)),td.y(isnan(td.h)),td.ts(isnan(td.h)),".b","MarkerSize",1);grid on;
title("Cleaned events");xlabel("x");ylabel("y");zlabel("t")
ylim([0 max(td.y)]);

ax(4)=subplot(2,3,4);
imagesc(log(eventCount_raw));colorbar;
title("Original events");
xlabel("x");ylabel("y");

ax(5)=subplot(2,3,5);
imagesc(log(eventCount_raw));colorbar;hold on
plot(hotpix_events(:,1),hotpix_events(:,2),".r");
plot(noisy_events(:,1),noisy_events(:,2),".g");
legend("Hot pixel","Noise");title("Labeled events");
xlabel("x");ylabel("y");

ax(6)=subplot(2,3,6);
imagesc(log(eventCount_cleaned));colorbar;
title("Cleaned events");
% ylim([0 max(td.y)]);
xlabel("x");ylabel("y");
linkaxes(ax, 'xy');

