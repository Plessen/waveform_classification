fs = 100e6;
A = 1;
p4_signal = type_P4(5, fs, A, fs  / 4, 64);
N = 1024;
p4_signal = p4_signal(1:1024);
sigma = 0.15;
[STFT,SST,VSST,~,~,~,~] = sst2_new(p4_signal,1 / sigma^2 / N ,N,1e-2);

displayZoomedImage(abs(STFT), [500, 250], 200, fs);
displayZoomedImage(abs(SST), [500, 250], 200, fs);
displayZoomedImage(abs(VSST), [500, 250], 200, fs);

function displayZoomedImage(image, zoomCenter, zoomSize, fs)
    % Function to display a 1024x1024 image with a zoomed-in rectangle using imagesc.
    %
    % Parameters:
    % - image: The input 1024x1024 matrix (image).
    % - zoomCenter: A 2-element vector specifying the [x, y] center of the zoom region.
    %               If empty, defaults to the center of the image.
    % - zoomSize: A scalar specifying the side length of the zoomed region. Defaults to 100.
    % - fs: Sampling frequency in Hz (to map time and frequency axes).
    %
    % Example usage:
    % displayZoomedImageWithImagesc(myImage, [512, 512], 150, 1000);

    % Input validation
    if nargin < 2 || isempty(zoomCenter)
        zoomCenter = [512, 512]; % Default to the center of the image
    end
    if nargin < 3 || isempty(zoomSize)
        zoomSize = 100; % Default size of the zoomed region
    end

    % Ensure the image is 1024x1024
    if size(image, 1) ~= 1024 || size(image, 2) ~= 1024
        error('The input image must be 1024x1024 in size.');
    end

    % Compute time and frequency axes
    time = (0:(1024-1)) / fs;  % Time axis
    freq = (0:(1024-1)) * (fs / 1024);  % Frequency axis (one-sided)

    % Calculate zoom region boundaries
    xStart = max(1, zoomCenter(1) - zoomSize / 2);
    xEnd = min(size(image, 2), zoomCenter(1) + zoomSize / 2);
    yStart = max(1, zoomCenter(2) - zoomSize / 2);
    yEnd = min(size(image, 1), zoomCenter(2) + zoomSize / 2);

    % Crop the zoomed region
    zoomedRegion = image(round(yStart):round(yEnd), round(xStart):round(xEnd));
    
    % Display the original image with a rectangle
    figure;

    % Plot the original image with time and frequency axes
    subplot(2, 1, 1);
    imagesc(time, freq, image); % Pass time and frequency as axes
    colormap('jet'); % Choose a colormap for better visualization
    colorbar;
    axis xy; % Ensure frequency axis increases upward
    %axis image; % Add this line to enforce square aspect ratio
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    set(gca, 'FontSize', 25); % Increase axis font size
    hold on;

    % Draw rectangle on the original image
    rectangle('Position', [(xStart - 1) / fs, (yStart - 1) * (fs / 1024), zoomSize / fs, zoomSize * (fs / 1024)], ...
              'EdgeColor', 'w', 'LineWidth', 2);
    
    % Display the zoomed region with time and frequency axes
    subplot(2, 1, 2);
    imagesc(linspace(time(xStart), time(xEnd), size(zoomedRegion, 2)), ...
            linspace(freq(round(yStart)), freq(round(yEnd)), size(zoomedRegion, 1)), ...
            zoomedRegion);
    colormap('jet'); % Use the same colormap for consistency
    colorbar;
    axis xy; % Ensure frequency axis increases upward
    %axis image; % Add this line to enforce square aspect ratio
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    set(gca, 'FontSize', 25); % Increase axis font size
    

    %set(gcf, 'Position', [100, 100, 800, 800]); % Square figure window (optional)
end

