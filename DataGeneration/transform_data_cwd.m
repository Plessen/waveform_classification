function resized_images = transform_data_cwd(signal, noisy_signal, N, image_size, real_noise_std, resize_method, transform, sigma)
    % Set sigma parameter for SST computation
    
    % Ensure signal and noisy_signal have the same length and pad if necessary
    signal = signal(1:min(N, length(signal)));
    noisy_signal = noisy_signal(1:min(N, length(noisy_signal)));
    original_size = length(signal);

    % Pad signals to the target length N
    noisy_signal = [noisy_signal, zeros(size(noisy_signal, 1), N - length(noisy_signal))];
    
    [cwd,~,~] = CWD(noisy_signal, 100e6, sigma);
    cwd = cwd(:, 1:original_size);

    resized_images.transform_noisy_resized = imresize(cwd, [image_size, image_size], resize_method, "Antialiasing", false);
end