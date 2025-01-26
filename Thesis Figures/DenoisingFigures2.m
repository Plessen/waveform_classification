fs = 100e6;
s = type_P1(4, fs, 1, fs / 4, 8);

%s = type_LFM(1024,3.8e6,1,3.8e6 / 4,3.8e6 / 24,"UP");
s = s(1:1024);
N = 1024;
SNR = -20:2:-18;
iterations = 10;
output_snr = zeros(3, length(SNR));
sigma = 0.16;
t = (1:1024) * 1 / fs;

for snr = 1:length(SNR)
    for i = 1:iterations
        noise = 1 / sqrt((10^(SNR(snr) / 10))) * 1/sqrt(2) * (randn(1, size(s, 2)) + 1i*randn(1, size(s,2)));
        %noise = 0;
        s_noise = s + noise;
        [STFT,SST,VSST,~,~,~,~] = sst2_new(s_noise,1 / sigma^2 / N ,N,std(real(noise)));
        %[WT, WSST, WSST2, fs, as, ~, ~, ~, ~, ~] = Wsst2_new(s_noise,std(real(noise)), 'cmor30-3',32);
        
        CWTopt=struct('gamma',eps,'type','gauss','mu',30,'s',3,'om',0,'nv',64,'freqscale','linear');
        [Tx, fs, Wx, as, Cw] = synsq_cwt_fw(t, s_noise, CWTopt.nv, CWTopt);
        gamma = est_riskshrink_thresh(Wx, 64);
        CWTopt=struct('gamma',std(real(noise)),'type','gauss','mu',30,'s',3,'om',0,'nv',64,'freqscale','linear');
        [Tx, fs, Wx, as, Cw] = synsq_cwt_fw(t, s_noise, CWTopt.nv, CWTopt);

        sinal_WSST = synsq_cwt_iw(Tx, fs, CWTopt).' * 1 / sqrt(pi);
        sinal_WSST = hilbert(sinal_WSST(:, 1).');
        sinal_SST = sum(SST);
        sinal_VSST = sum(VSST);
        %sinal_WSST2 = iwsst(WSST2, "amor", WaveletParameters=[sqrt(30 / 2) 3]) * sqrt(pi);
        %sinal_WSST2 = sinal_WSST2.';
        output_snr(1, snr) = output_snr(1, snr) + 10*log10(N / sum(abs(sinal_SST- s).^2));
        output_snr(2, snr) = output_snr(2, snr) + 10*log10(N / sum(abs(sinal_VSST- s).^2));
        output_snr(3, snr) = output_snr(3, snr) + 10*log10(N / sum(abs(sinal_WSST- s).^2));

        fprintf("%f \n", 1 / N * sum(abs(s - sinal_SST)))
        fprintf("%f \n", 1 / N * sum(abs(s - sinal_VSST)))
        fprintf("%f \n", 1 / N * sum(abs(s - sinal_WSST)))
    end
end

output_snr = output_snr ./ iterations;

figure;
plot(SNR, output_snr(1,:));
hold on;
plot(SNR, output_snr(2, :));
hold on;
plot(SNR, output_snr(3, :));
legend("SST", "VSST", "WSST2");

figure();
imagesc(angle(SST));
figure();
imagesc(angle(VSST));

%SST = imresize(SST, [128, 128], 'nearest');
%VSST = imresize(VSST, [128, 128], 'nearest');



