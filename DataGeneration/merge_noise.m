function [noisy_signal, real_noise_std] = merge_noise(s, snr_db)
    noise = 1 / sqrt((10^(snr_db / 10))) * 1/sqrt(2) * (randn(1, size(s, 2)) + 1i*randn(1, size(s,2)));
    noisy_signal = s + noise;
    real_noise_std = std(real(noise));
end