function resized_images = transform_data(signal, noisy_signal, N, image_size, real_noise_std, resize_method, transform)
    % Set sigma parameter for SST computation
    sigma = 0.06; %This offers an ok frequency resolution, but the time resolution offers better discrimination (changing this will definitly have impact on model performance)

    % Ensure signal and noisy_signal have the same length and pad if necessary
    signal = signal(1:min(N, length(signal)));
    noisy_signal = noisy_signal(1:min(N, length(noisy_signal)));
    original_size = length(signal);

    % Pad signals to the target length N
    noisy_signal = [noisy_signal, zeros(size(noisy_signal, 1), N - length(noisy_signal))];
    signal = [signal, zeros(size(signal, 1), N - length(signal))];

    % Compute SST and VSST for noisy signal
    [~, SST_noisy, VSST_noisy, ~, ~, ~, ~] = sst2_new(noisy_signal, 1 / sigma^2 / N, N, real_noise_std * 1 / sqrt(2));
    SST_noisy = SST_noisy(1:N/2, 1:original_size);
    VSST_noisy = VSST_noisy(1:N/2, 1:original_size);

    % Compute SST and VSST for clean signal
    [~, SST_clean, VSST_clean, ~, ~, ~, ~] = sst2_new(signal, 1 / sigma^2 / N, N, 0);
    SST_clean = SST_clean(1:N/2, 1:original_size);
    VSST_clean = VSST_clean(1:N/2, 1:original_size);
    
    if transform == "SST"
        resized_images.transform_resized = imresize(SST_clean, [image_size, image_size], resize_method, "Antialiasing", false);
        resized_images.transform_noisy_resized = imresize(SST_noisy, [image_size, image_size], resize_method, "Antialiasing", false);
    else
        resized_images.transform_resized = imresize(VSST_clean, [image_size, image_size], resize_method, "Antialiasing", false);
        resized_images.transform_noisy_resized = imresize(VSST_noisy, [image_size, image_size], resize_method, "Antialiasing", false);
    end
end


