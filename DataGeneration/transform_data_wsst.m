function resized_images = transform_data_wsst(signal, noisy_signal, N, image_size, real_noise_std, resize_method, transform, sigma)
    % Set sigma parameter for SST computation
    
    % Ensure signal and noisy_signal have the same length and pad if necessary
    signal = signal(1:min(N, length(signal)));
    noisy_signal = noisy_signal(1:min(N, length(noisy_signal)));
    original_size = length(signal);

    % Pad signals to the target length N
    noisy_signal = [noisy_signal, zeros(size(noisy_signal, 1), N - length(noisy_signal))];
    
    CWTopt=struct('gamma',eps,'type','morlet','mu',30,'s',3,'om',0,'nv',64,'freqscale','linear');
    [Tx, ~, ~, ~, ~] = synsq_cwt_fw((1:N), noisy_signal, CWTopt.nv, CWTopt);
    Tx = Tx(:, 1:original_size);

    resized_images.transform_resized = imresize(Tx, [image_size, image_size], resize_method, "Antialiasing", false);
end