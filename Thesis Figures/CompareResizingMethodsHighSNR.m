fs = 100e6;
A = 1;
N = 1024;
addpath("../DataGeneration");
addpath("../DataGeneration/Transforms");
addpath("../DataGeneration/Waveforms");
addpath("C:\Users\User\Documents\Thesis\Thesis\FSSTn-master\FSSTn-master")
wav_T3 = type_T3(N, fs, A, fs / 6, 2, fs / 10);
wav_Costas = type_Costas(N, fs, A, fs / 24, getCostasHopingSequence(6));
wav_Frank = type_Frank(5, fs, A, fs / 4, 8);
wav_T4 = type_T4(N, fs, A, fs / 5, 2, fs / 10);
wav_Barker = type_Barker(10, fs, A, fs / 5, [0 0 0 0 0 1 1 0 0 1 0 1 0]*pi);
wav_P1 = type_P1(5, fs, A, fs / 4, 8);
wav_Costas = wav_Costas(1:1024);
wav_Frank = wav_Frank(1:1024);

sigma = 0.08;
[~, SST_T3, ~, ~, ~, ~, g] = sst2_new(wav_T3, 1 / sigma^2 / N, N, 0);
[~, SST_Costas, ~, ~, ~, ~, ~] = sst2_new(wav_Costas, 1 / sigma^2 / N, N, 0);
[~, SST_Frank, ~, ~, ~, ~, ~] = sst2_new(wav_Frank, 1 / sigma^2 / N, N, 0);
[~, SST_T4, ~, ~, ~, ~, ~] = sst2_new(wav_T4, 1 / sigma^2 / N, N, 0);
[~, SST_Barker, ~, ~, ~, ~, ~] = sst2_new(wav_Barker, 1 / sigma^2 / N, N, 0);
[~, SST_P1, ~, ~, ~, ~, ~] = sst2_new(wav_P1, 1 / sigma^2 / N, N, 0);
[~,SST_T3_V1,~,~,~,~,~,~,~,~,~,~,~, g1] = sstn(wav_T3,0,sigma);

% Resize the images to 128x128 pixels
SST_T3_nearest = imresize(SST_T3(1:512, :), [128, 128], 'nearest');
SST_T3_bilinear = imresize(SST_T3(1:512, :), [128, 128], 'bilinear', 'Antialiasing', false);

SST_Costas_nearest = imresize(SST_Costas(1:512, :), [128, 128], 'nearest');
SST_Costas_bilinear = imresize(SST_Costas(1:512, :), [128, 128], 'bilinear', 'Antialiasing', false);

SST_Frank_nearest = imresize(SST_Frank(1:512, :), [128, 128], 'nearest');
SST_Frank_bilinear = imresize(SST_Frank(1:512, :), [128, 128], 'bilinear', 'Antialiasing', false);

SST_T4_nearest = imresize(SST_T4(1:512, :), [128, 128], 'nearest');
SST_T4_bilinear = imresize(SST_T4(1:512, :), [128, 128], 'bilinear', 'Antialiasing', false);

SST_Barker_nearest = imresize(SST_Barker(1:512, :), [128, 128], 'nearest');
SST_Barker_bilinear = imresize(SST_Barker(1:512, :), [128, 128], 'bilinear', 'Antialiasing', false);

SST_P1_nearest = imresize(SST_P1(1:512, :), [128, 128], 'nearest');
SST_P1_bilinear = imresize(SST_P1(1:512, :), [128, 128], 'bilinear', 'Antialiasing', false);
% Plot the results side by side
figure('Position', [100, 100, 900, 500], 'Units', 'normalized');
tiledlayout(2, 4, 'TileSpacing', 'tight', 'Padding', 'none');

%nexttile; imagesc(abs(SST_T3_nearest)); axis image off; colormap jet; title('T3 Nearest', 'FontSize', 16);
%nexttile; imagesc(abs(SST_T3_bilinear)); axis image off; colormap jet; title('T3 Bilinear', 'FontSize', 16);

nexttile; imagesc(abs(SST_Costas_nearest)); axis image off; colormap jet; title('Costas Nearest', 'FontSize', 32);
nexttile; imagesc(abs(SST_Costas_bilinear)); axis image off; colormap jet; title('Costas Bilinear', 'FontSize', 32);

nexttile; imagesc(abs(SST_Frank_nearest)); axis image off; colormap jet; title('Frank Nearest', 'FontSize', 32);
nexttile; imagesc(abs(SST_Frank_bilinear)); axis image off; colormap jet; title('Frank Bilinear', 'FontSize', 32);

nexttile; imagesc(abs(SST_T4_nearest)); axis image off; colormap jet; title('T4 Nearest', 'FontSize', 32);
nexttile; imagesc(abs(SST_T4_bilinear)); axis image off; colormap jet; title('T4 Bilinear', 'FontSize', 32);

nexttile; imagesc(abs(SST_Barker_nearest)); axis image off; colormap jet; title('Barker Nearest', 'FontSize', 32);
nexttile; imagesc(abs(SST_Barker_bilinear)); axis image off; colormap jet; title('Barker Bilinear', 'FontSize', 32);

%nexttile; imagesc(abs(SST_P1_nearest)); axis image off; colormap jet; title('P1 Nearest', 'FontSize', 16);
%nexttile; imagesc(abs(SST_P1_bilinear)); axis image off; colormap jet; title('P1 Bilinear', 'FontSize', 16);
