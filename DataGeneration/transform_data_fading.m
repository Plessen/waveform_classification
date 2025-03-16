function resized_images = transform_data_fading(signal, SNR, N, image_size, resize_method, transform, sigma,... 
    numPaths_range, pathDelay_range, pathGain_range, Kfactor_range, dopplerRange, fs)

    PathNum = randi(numPaths_range,1,1);
    pathDelayConstant = randi(pathDelay_range,1,1);
    avgPathGainsConstant = randi(pathGain_range,1,1);
    
    pathDelays = [0:PathNum-1].*pathDelayConstant*1e-9;
    avgPathGains = -1*[0:PathNum-1].*avgPathGainsConstant;
    K = randi(Kfactor_range,1,1);
    fdopMax = randi(dopplerRange,1,1);
    
    ricianChan = comm.RicianChannel(...
        'SampleRate',fs,...
        'PathDelays',pathDelays,...
        'AveragePathGains',avgPathGains,...
        'KFactor',K,...
        'MaximumDopplerShift',fdopMax, ...
        'ChannelFiltering', true);


    fadingOutput = ricianChan(signal.');
    output = awgn(fadingOutput.', SNR, 'measured');

    
    % Ensure signal and noisy_signal have the same length and pad if necessary
    output = signal(1:min(N, length(output)));
    original_size = length(output);

    signal = [output, zeros(size(output, 1), N - length(output))];

    if transform == "SST" || transform == "VSST"
        [~, SST_noisy, VSST_noisy, ~, ~, ~, ~] = sst2_new(signal, 1 / sigma^2 / N, N, 0);
        SST_noisy = SST_noisy(1:N/2, 1:original_size);
        VSST_noisy = VSST_noisy(1:N/2, 1:original_size);

        if transform == "SST"
            resized_images.transform_resized = imresize(SST_noisy, [image_size, image_size], resize_method, "Antialiasing", false);
        else
            resized_images.transform_resized = imresize(VSST_noisy, [image_size, image_size], resize_method, "Antialiasing", false);
        end

    elseif transform == "kaiser"
        padding = zeros((1024 - 256) / 2, 1);
        window = [padding; kaiser(256,10); padding];
        [SST_noisy,~,~] = fsst(noisy_signal, fs, window,'yaxis');
        SST_noisy = SST_noisy(N/2:N, 1:original_size);
        resized_images.transform_resized = imresize(SST_noisy, [image_size, image_size], resize_method, "Antialiasing", false);
    
    elseif transform == "CWD"
        [cwd,~,~] = CWD(noisy_signal, fs, sigma);
        cwd = cwd(:, 1:original_size);
        resized_images.transform_resized = imresize(cwd, [image_size, image_size], resize_method, "Antialiasing", false);
    end

end