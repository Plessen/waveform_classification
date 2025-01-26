s = type_P1(4, 3.8e6, 1, 3.8e6 / 4, 8);
s = s(1:1024);
N = 1024;
SNR =-0:1:0;
iterations = 5;
output_snr = zeros(3, length(SNR));
sigma = 0.15;
for snr = 1:length(SNR)
    for i = 1:iterations
        noise = 1 / sqrt((10^(SNR(snr) / 10))) * 1/sqrt(2) * (randn(1, size(s, 2)) + 1i*randn(1, size(s,2)));
        %noise = 0;
        s_noise = s + noise;
        threshold = std(real(noise));
        %threshold = 0;
        if SNR(snr) > -11
            [STFT,SST,VSST,~,~,~,~] = sst2_new(s_noise,1 / sigma^2 / N ,N,threshold);
            [WT, WSST, WSST2, fs, as, ~, ~, ~, ~, ~] = Wsst2_new(s_noise,std(real(noise)), 'cmor6-2',128);
            %WSST2 = wsst(real(s_noise));
        else
            [STFT,SST,VSST,~,~,~,~] = sst2_new(s_noise,1 / sigma^2 / N ,N,threshold);
            [WT, WSST, WSST2, fs, as, ~, ~, ~, ~, ~] = Wsst2_new(s_noise,std(real(noise)) * 0.1, 'cmor6-7',128);
        end
        %estimate = mad(abs())

        sinal_SST = sum(SST);
        sinal_VSST = sum(VSST);
        sinal_WSST2 = iwsst(WSST2);
        sinal_WSST2 = sinal_WSST2.';
        output_snr(1, snr) = output_snr(1, snr) + 10*log10(N / sum(abs(sinal_SST - s).^2));
        output_snr(2, snr) = output_snr(2, snr) + 10*log10(N / sum(abs(sinal_VSST - s).^2));
        output_snr(3, snr) = output_snr(3, snr) + 10*log10(N / sum(abs(sinal_WSST2 - s).^2));
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
imagesc(abs(SST));
figure();
imagesc(abs(VSST));

%SST = imresize(SST, [128, 128], 'nearest');
%VSST = imresize(VSST, [128, 128], 'nearest');

figure();
imagesc(abs(SST));
figure();
imagesc(abs(VSST));


figure();
imagesc(abs(WSST2));

figure();
WSST2 = WSST2.';
freqmin = 1./as(end) ;
freqmax = 1./as(1) ;
freqtic = linspace(freqmin, freqmax, 1024) ;
tfr2 = zeros(size(WSST2,1), length(freqtic)) ;
for jj = 1: size(tfr2,1)
    tmp = interp1(1./as, WSST2(jj,:), freqtic, 'linear') ;
    tfr2(jj, :) = tmp ;
end


imagesc(abs(tfr2.').^2);


