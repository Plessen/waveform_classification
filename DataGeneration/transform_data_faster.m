function resized_images = transform_data_faster(signal, SNR, N, image_size, resize_method, transform, sigma, fs)
    [noisy_signal, ~] = merge_noise(signal, SNR);
    
    noisy_signal = noisy_signal(1:min(N, length(noisy_signal)));
    original_size = length(noisy_signal);

    noisy_signal = [noisy_signal, zeros(size(noisy_signal, 1), N - length(noisy_signal))];

    if transform == "SST" || transform == "VSST" || transform == "STFT"
        n = 1024;
        t = -0.5:1/n:0.5-1/n;t=t';
        g =  1/sigma*exp(-pi/sigma^2*t.^2);
        
        SST_noisy = fsst(noisy_signal, fs, g, 'yaxis');
        %[STFT, SST_noisy, VSST_noisy, ~, ~, ~, ~] = sst2_new(noisy_signal, 1 / sigma^2 / N, N, 0);
        SST_noisy = SST_noisy(N/2:N, 1:original_size);
        %VSST_noisy = VSST_noisy(1:N/2, 1:original_size);

        if transform == "SST"
            resized_images.transform_resized = imresize(SST_noisy, [image_size, image_size], resize_method, "Antialiasing", false);
        elseif transform == "VSST"
            resized_images.transform_resized = imresize(SST_noisy, [image_size, image_size], resize_method, "Antialiasing", false);
        else
            [STFT, ~, ~, ~, ~, ~, ~] = sst2_new(noisy_signal, 1 / sigma^2 / N, N, 0);
            STFT = STFT(1:N/2, 1:original_size);
            resized_images.transform_resized = imresize(STFT, [image_size, image_size], resize_method, "Antialiasing", false);
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