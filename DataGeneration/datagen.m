fs = 100e6;
image_size = 128;
A = 1;
waveforms = {'LFM','Costas','Barker','Frank','P1','P2','P3','P4'};
signals_per_SNR = 1000;
SNR = -14:2:-4;

generate_nearest = true;
generate_bilinear = false;
generate_bicubic = false;
generate_lanczos2 = false;
generate_lanczos3 = false;

global input_train_nearest_SST input_train_nearest_noisy_SST input_train_nearest_VSST input_train_nearest_noisy_VSST;
global input_train_bilinear_SST input_train_bilinear_noisy_SST input_train_bilinear_VSST input_train_bilinear_noisy_VSST;
global input_train_bicubic_SST input_train_bicubic_noisy_SST input_train_bicubic_VSST input_train_bicubic_noisy_VSST;
global input_train_lanczos2_SST input_train_lanczos2_noisy_SST input_train_lanczos2_VSST input_train_lanczos2_noisy_VSST;
global input_train_lanczos3_SST input_train_lanczos3_noisy_SST input_train_lanczos3_VSST input_train_lanczos3_noisy_VSST;

if generate_nearest
    input_train_nearest_SST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_nearest_noisy_SST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_nearest_VSST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_nearest_noisy_VSST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
end

if generate_bilinear
    input_train_bilinear_SST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_bilinear_noisy_SST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_bilinear_VSST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_bilinear_noisy_VSST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
end

if generate_bicubic
    input_train_bicubic_SST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_bicubic_noisy_SST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_bicubic_VSST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_bicubic_noisy_VSST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
end

if generate_lanczos2
    input_train_lanczos2_SST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_lanczos2_noisy_SST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_lanczos2_VSST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_lanczos2_noisy_VSST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
end

if generate_lanczos3
    input_train_lanczos3_SST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_lanczos3_noisy_SST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_lanczos3_VSST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
    input_train_lanczos3_noisy_VSST = zeros(signals_per_SNR * length(waveforms) * length(SNR), image_size, image_size);
end


output_train_data = zeros(signals_per_SNR * length(waveforms) * length(SNR), length(waveforms));

data_index = 1;
for snr_index = 1:length(SNR)
    for waveform_index = 1:length(waveforms)
        waveform  = waveforms{waveform_index};
        
        switch waveform
            case 'LFM'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/6,fs/4, signals_per_SNR);
                fc=fc(randperm(signals_per_SNR)); %random permutation
                B = linspace(fs/20, fs/16, signals_per_SNR);
                B = B(randperm(signals_per_SNR)); %random permutation
                N = linspace(512,1024,signals_per_SNR);
                N=round(N(randperm(signals_per_SNR))); %random permutation
                sweepDirections = {'Up','Down'};

                output_vector = zeros(1, length(waveforms));
                output_vector(waveform_index) = 1;
                for idx = 1:signals_per_SNR
                    wav = type_LFM(N(idx),fs,A,fc(idx),B(idx),sweepDirections{randi(2)});
                    [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                    resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std);
                    output_train_data = [output_train_data; output_vector];
                    assign_images(resized_images, data_index, generate_nearest, generate_bilinear, generate_bicubic, generate_lanczos2, generate_lanczos3);
                    data_index = data_index  + 1; % DONT FORGET
                end

            case 'Costas'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [3, 4, 5, 6];
                fcmin = linspace(fs/24,fs/20,signals_per_SNR);
                fcmin=fcmin(randperm(signals_per_SNR));
                N = linspace(512,1024,signals_per_SNR);
                N=round(N(randperm(signals_per_SNR)));
                    
                output_vector = zeros(1, length(waveforms));
                output_vector(waveform_index) = 1;

                for idx = 1:signals_per_SNR
                    NumHop = randperm(Lc(randi(4)));
                    wav = type_Costas(N(idx), fs, A, fcmin(idx), NumHop);
                    [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                    resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std);
                    output_train_data = [output_train_data; output_vector];
                    assign_images(resized_images, data_index, generate_nearest, generate_bilinear, generate_bicubic, generate_lanczos2, generate_lanczos3);
                    data_index = data_index  + 1; % DONT FORGET
                end
            
            case 'Barker'
                disp(['Generating ',waveform, ' waveform ...']);
                Lc = [7,11,13];
                fc = linspace(fs/6,fs/4,signals_per_SNR);
                fc=fc(randperm(signals_per_SNR));
                Ncc = 10:13;
                    
                output_vector = zeros(1, length(waveforms));
                output_vector(waveform_index) = 1;

                for idx = 1:signals_per_SNR
                    Bar = Lc(randi(3));
                    if Bar == 7
                        phaseCode = [0 0 0 1 1 0 1]*pi;
                    elseif Bar == 11
                        phaseCode = [0 0 0 1 1 1 0 1 1 0 1]*pi;
                    elseif Bar == 13
                        phaseCode = [0 0 0 0 0 1 1 0 0 1 0 1 0]*pi;
                    end
                    wav = type_Barker(Ncc, fs, A, fc(idx), phaseCode);
                    [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                    resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std);
                    output_train_data = [output_train_data; output_vector];
                    assign_images(resized_images, data_index, generate_nearest, generate_bilinear, generate_bicubic, generate_lanczos2, generate_lanczos3);
                    data_index = data_index  + 1; % DONT FORGET
                end

            case 'Frank'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/4,fs/3,signals_per_SNR);
                fc=fc(randperm(signals_per_SNR));
                Ncc = [3,4,5];
                M = [6, 7, 8];
                    
                output_vector = zeros(1, length(waveforms));
                output_vector(waveform_index) = 1;

                for idx = 1:signals_per_SNR
                    wav = type_Frank(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                    resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std);
                    output_train_data = [output_train_data; output_vector];
                    assign_images(resized_images, data_index, generate_nearest, generate_bilinear, generate_bicubic, generate_lanczos2, generate_lanczos3);
                    data_index = data_index  + 1; % DONT FORGET
                end

            case 'P1'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/4,fs/3,signals_per_SNR);
                fc=fc(randperm(signals_per_SNR));
                Ncc = [3,4,5];
                M = [6, 7, 8];
                    
                output_vector = zeros(1, length(waveforms));
                output_vector(waveform_index) = 1;

                for idx = 1:signals_per_SNR
                    wav = type_P1(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                    [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                    resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std);
                    output_train_data = [output_train_data; output_vector];
                    assign_images(resized_images, data_index, generate_nearest, generate_bilinear, generate_bicubic, generate_lanczos2, generate_lanczos3);
                    data_index = data_index  + 1; % DONT FORGET
                end

            case 'P2'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/4,fs/3,signals_per_SNR);
                fc=fc(randperm(signals_per_SNR));
                Ncc = [3,4,5];
                M = [6, 8];
                    
                output_vector = zeros(1, length(waveforms));
                output_vector(waveform_index) = 1;

                for idx = 1:signals_per_SNR
                    wav = type_P2(Ncc(randi(3)), fs, A, fc(idx), M(randi(2)));
                    [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                    resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std);
                    output_train_data = [output_train_data; output_vector];
                    assign_images(resized_images, data_index, generate_nearest, generate_bilinear, generate_bicubic, generate_lanczos2, generate_lanczos3);
                    data_index = data_index  + 1; % DONT FORGET
                end

            case 'P3'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/4,fs/3,signals_per_SNR);
                fc=fc(randperm(signals_per_SNR));
                Ncc = [3,4,5];
                p = [36, 49, 64];
                    
                output_vector = zeros(1, length(waveforms));
                output_vector(waveform_index) = 1;

                for idx = 1:signals_per_SNR
                    wav = type_P3(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                    resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std);
                    output_train_data = [output_train_data; output_vector];
                    assign_images(resized_images, data_index, generate_nearest, generate_bilinear, generate_bicubic, generate_lanczos2, generate_lanczos3);
                    data_index = data_index  + 1; % DONT FORGET
                end

            case 'P4'
                disp(['Generating ',waveform, ' waveform ...']);
                fc = linspace(fs/4,fs/3,signals_per_SNR);
                fc=fc(randperm(signals_per_SNR));
                Ncc = [3,4,5];
                p = [36, 49, 64];
                    
                output_vector = zeros(1, length(waveforms));
                output_vector(waveform_index) = 1;

                for idx = 1:signals_per_SNR
                    wav = type_P4(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                    [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                    resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std);
                    output_train_data = [output_train_data; output_vector];
                    assign_images(resized_images, data_index, generate_nearest, generate_bilinear, generate_bicubic, generate_lanczos2, generate_lanczos3);
                    data_index = data_index  + 1; % DONT FORGET
                end
            otherwise
                disp('FINISHED')
        end
    end %end waveform_index
end %end snr_index

if generate_nearest
    save('input_train_nearest_SST.mat', 'input_train_nearest_SST', '-v7.3');
    save('input_train_nearest_noisy_SST.mat', 'input_train_nearest_noisy_SST', '-v7.3');
    save('input_train_nearest_VSST.mat', 'input_train_nearest_VSST', '-v7.3');
    save('input_train_nearest_noisy_VSST.mat', 'input_train_nearest_noisy_VSST', '-v7.3');
end

if(generate_bilinear)
    save('input_train_bilinear_SST.mat', 'input_train_bilinear_SST', '-v7.3');
    save('input_train_bilinear_noisy_SST.mat', 'input_train_bilinear_noisy_SST', '-v7.3');
    save('input_train_bilinear_VSST.mat', 'input_train_bilinear_VSST', '-v7.3');
    save('input_train_bilinear_noisy_VSST.mat', 'input_train_bilinear_noisy_VSST', '-v7.3');
end

if(generate_bicubic)
    save('input_train_bicubic_SST.mat', 'input_train_bicubic_SST', '-v7.3');
    save('input_train_bicubic_noisy_SST.mat', 'input_train_bicubic_noisy_SST', '-v7.3');
    save('input_train_bicubic_VSST.mat', 'input_train_bicubic_VSST', '-v7.3');
    save('input_train_bicubic_noisy_VSST.mat', 'input_train_bicubic_noisy_VSST', '-v7.3');
end

if(generate_lanczos2)
    save('input_train_lanczos2_SST.mat', 'input_train_lanczos2_SST', '-v7.3');
    save('input_train_lanczos2_noisy_SST.mat', 'input_train_lanczos2_noisy_SST', '-v7.3');
    save('input_train_lanczos2_VSST.mat', 'input_train_lanczos2_VSST', '-v7.3');
    save('input_train_lanczos2_noisy_VSST.mat', 'input_train_lanczos2_noisy_VSST', '-v7.3');
end

if(generate_lanczos3)
    save('input_train_lanczos3_SST.mat', 'input_train_lanczos3_SST', '-v7.3');
    save('input_train_lanczos3_noisy_SST.mat', 'input_train_lanczos3_noisy_SST', '-v7.3');
    save('input_train_lanczos3_VSST.mat', 'input_train_lanczos3_VSST', '-v7.3');
    save('input_train_lanczos3_noisy_VSST.mat', 'input_train_lanczos3_noisy_VSST', '-v7.3');
end

% Save output data
save('output_train_data.mat', 'output_train_data', '-v7.3');
disp('Data generation and saving complete.');

function assign_images(resized_images, data_index, generate_nearest, generate_bilinear, generate_bicubic, generate_lanczos2, generate_lanczos3)
    global input_train_nearest_SST input_train_nearest_noisy_SST input_train_nearest_VSST input_train_nearest_noisy_VSST;
    global input_train_bilinear_SST input_train_bilinear_noisy_SST input_train_bilinear_VSST input_train_bilinear_noisy_VSST;
    global input_train_bicubic_SST input_train_bicubic_noisy_SST input_train_bicubic_VSST input_train_bicubic_noisy_VSST;
    global input_train_lanczos2_SST input_train_lanczos2_noisy_SST input_train_lanczos2_VSST input_train_lanczos2_noisy_VSST;
    global input_train_lanczos3_SST input_train_lanczos3_noisy_SST input_train_lanczos3_VSST input_train_lanczos3_noisy_VSST;

    if(generate_nearest)
        input_train_nearest_SST(data_index, :, :) = resized_images.SST_resized_nearest;
        input_train_nearest_noisy_SST(data_index, :, :) = resized_images.SST_noisy_resized_nearest;
        input_train_nearest_VSST(data_index, :, :) = resized_images.VSST_resized_nearest;
        input_train_nearest_noisy_VSST(data_index, :, :) = resized_images.VSST_noisy_resized_nearest;
    end

    if(generate_bilinear)
        input_train_bilinear_SST(data_index, :, :) = resized_images.SST_resized_bilinear;
        input_train_bilinear_noisy_SST(data_index, :, :) = resized_images.SST_noisy_resized_bilinear;
        input_train_bilinear_VSST(data_index, :, :) = resized_images.VSST_resized_bilinear;
        input_train_bilinear_noisy_VSST(data_index, :, :) = resized_images.VSST_noisy_resized_bilinear;
    end

    if(generate_bicubic)
        input_train_bicubic_SST(data_index, :, :) = resized_images.SST_resized_bicubic;
        input_train_bicubic_noisy_SST(data_index, :, :) = resized_images.SST_noisy_resized_bicubic;
        input_train_bicubic_VSST(data_index, :, :) = resized_images.VSST_resized_bicubic;
        input_train_bicubic_noisy_VSST(data_index, :, :) = resized_images.VSST_noisy_resized_bicubic;
    end

    if(generate_lanczos2)
        input_train_lanczos2_SST(data_index, :, :) = resized_images.SST_resized_lanczos2;
        input_train_lanczos2_noisy_SST(data_index, :, :) = resized_images.SST_noisy_resized_lanczos2;
        input_train_lanczos2_VSST(data_index, :, :) = resized_images.VSST_resized_lanczos2;
        input_train_lanczos2_noisy_VSST(data_index, :, :) = resized_images.VSST_noisy_resized_lanczos2;
    end

    if(generate_lanczos3)
        input_train_lanczos3_SST(data_index, :, :) = resized_images.SST_resized_lanczos3;
        input_train_lanczos3_noisy_SST(data_index, :, :) = resized_images.SST_noisy_resized_lanczos3;
        input_train_lanczos3_VSST(data_index, :, :) = resized_images.VSST_resized_lanczos3;
        input_train_lanczos3_noisy_VSST(data_index, :, :) = resized_images.VSST_noisy_resized_lanczos3;
    end
end