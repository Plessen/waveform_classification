function resized_images = transform_data_fading(signal, SNR, N, image_size, resize_method, transform, sigma,... 
    numPaths_range, pathDelay_range, pathGain_range, Kfactor_range, fs, strategy, number_antennas)

    
    output = fading_strategy(signal, SNR, strategy, number_antennas, numPaths_range, pathDelay_range, pathGain_range, Kfactor_range, fs);
    
    % Ensure signal and noisy_signal have the same length and pad if necessary
    output = output(1:min(N, length(output)));
    original_size = length(output);

    output = [output; zeros(N - length(output), 1)];
    if transform == "SST" || transform == "VSST" || transform == "STFT"
        
        if transform == "SST"
            n = 1024;
            t = -0.5:1/n:0.5-1/n;t=t';
            g =  1/sigma*exp(-pi/sigma^2*t.^2);
            SST_noisy = fsst(output, fs, g, 'yaxis');
            SST_noisy = SST_noisy(N/2:N, 1:original_size);
            resized_images.transform_resized = imresize(SST_noisy, [image_size, image_size], resize_method, "Antialiasing", false);
        elseif transform == "VSST"
            [~, ~, VSST_noisy, ~, ~, ~, ~] = sst2_new(output, 1 / sigma^2 / N, N, 0);
            VSST_noisy = VSST_noisy(1:N/2, 1:original_size);
            resized_images.transform_resized = imresize(VSST_noisy, [image_size, image_size], resize_method, "Antialiasing", false);
        else
            [STFT, ~, ~, ~, ~, ~, ~] = sst2_new(output, 1 / sigma^2 / N, N, 0);
            STFT = STFT(1:N/2, 1:original_size);
            resized_images.transform_resized = imresize(STFT, [image_size, image_size], resize_method, "Antialiasing", false);
        end

    elseif transform == "kaiser"
        padding = zeros((1024 - 256) / 2, 1);
        window = [padding; kaiser(256,10); padding];
        [SST_noisy,~,~] = fsst(output, fs, window,'yaxis');
        SST_noisy = SST_noisy(N/2:N, 1:original_size);
        resized_images.transform_resized = imresize(SST_noisy, [image_size, image_size], resize_method, "Antialiasing", false);
    
    elseif transform == "CWD"
        [cwd,~,~] = CWD(output, fs, sigma);
        cwd = cwd(:, 1:original_size);
        resized_images.transform_resized = imresize(cwd, [image_size, image_size], resize_method, "Antialiasing", false);
    end

end