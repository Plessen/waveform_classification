function noisy_signal = fading_strategy_flat(signal, SNR, strategy, Nr, numPaths_range, pathDelay_range, pathGain_range, Kfactor_range, fs, fc)
    N = length(signal); % Length of the input signal
    signal = signal(:);

    if strategy == "cooperative"
        received_signal = zeros(N, Nr);
        
        for i = 1:Nr
            % Generate per-node fading channel
            K = randi(Kfactor_range);  % Random K per antenna
        
            % LOS and NLOS components
            theta = 2 * pi * rand();
            h_LOS = sqrt(K / (K + 1)) * exp(1j * theta);
            h_NLOS = sqrt(1 / (2 * (K + 1))) * (randn() + 1j * randn());
        
            h_i = h_LOS + h_NLOS;
        
            % Apply fading and noise
            received_signal(:, i) = awgn(signal * h_i, SNR, 'measured');
        end
        
        aligned_signals = zeros(N, Nr);
        aligned_signals(:, 1) = received_signal(:, 1);
        for i = 2:Nr
            [r, ~] = xcorr(received_signal(:, 1), received_signal(:, i));
            phi_est = angle(r(N));
            aligned_signals(:, i) = received_signal(:, i) * exp(1i * phi_est);
        end

        noisy_signal = sum(aligned_signals, 2);
    elseif strategy == "independent"
        K = randi(Kfactor_range, 1, 1);
        theta = 2 * pi * rand(1, Nr);
        h_LOS = sqrt(K / (K + 1)) * exp(1j * theta);
        h_NLOS = sqrt(1 / (2 * (K + 1))) * (randn(1, Nr) + 1j * randn(1, Nr));
        h = h_LOS + h_NLOS;  % 1 x Nr complex gains
        received_signal = signal .* h;
        received_signal = awgn(received_signal, SNR, 'measured');

        aligned_signals = zeros(N, Nr);
        aligned_signals(:, 1) = received_signal(:, 1);
        for i = 2:Nr
            [r, ~] = xcorr(received_signal(:, 1), received_signal(:, i));
            phi_est = angle(r(N));
            aligned_signals(:, i) = received_signal(:, i) * exp(1i*phi_est);
        end
        noisy_signal = sum(aligned_signals, 2);

    elseif strategy  == "selection"
        K = randi(Kfactor_range, 1, 1);
        theta = 2 * pi * rand(1, Nr);
        h_LOS = sqrt(K / (K + 1)) * exp(1j * theta);
        h_NLOS = sqrt(1 / (2 * (K + 1))) * (randn(1, Nr) + 1j * randn(1, Nr));
        h = h_LOS + h_NLOS;  % 1 x Nr complex gains
        received_signal = signal .* h;
        received_signal = awgn(received_signal, SNR, 'measured');
        
         [~, idx] = max(sum(abs(received_signal).^2, 1));
         noisy_signal = received_signal(:, idx);

    elseif strategy == "maximum"
        K = randi(Kfactor_range, 1, 1);
        theta = 2 * pi * rand(1, Nr);
        h_LOS = sqrt(K / (K + 1)) * exp(1j * theta);
        h_NLOS = sqrt(1 / (2 * (K + 1))) * (randn(1, Nr) + 1j * randn(1, Nr));
        h = h_LOS + h_NLOS;  % 1 x Nr complex gains
        received_signal = signal .* h;
        received_signal = awgn(received_signal, SNR, 'measured');

        noisy_signal = sum(conj(H) .* received_signal, 2);

    end
end