function [] = cluster_parallel_data_gen(signals_per_SNR, resize_method, transform, num_workers, seed, train, sigma)
    
    %pool = initParPool();
    
    output_dir = "./data";
    assert(exist(output_dir, "dir"), "The output directory does not exist");

    addpath("./Waveforms");
    addpath("./Transforms");
    addpath("./data");
    fs = 100e6;
    image_size = 128;
    A = 1;
    waveforms = {'LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4'};
    SNR = -14:2:-4;
    pool = parpool(num_workers);
    total_signals_per_SNR = signals_per_SNR * length(waveforms);

    if train
        prefix = 'input_train_';
    else
        prefix = 'input_test_';
    end

    for snr_index = 1:length(SNR)
         prefix_clean = fullfile(output_dir, [prefix resize_method '_' transform '_' num2str(snr_index) '_' 'sigma' '_' num2str(sigma) '.h5']);
         h5create(prefix_clean, '/clean_images/images_real', [image_size, image_size, total_signals_per_SNR], 'Datatype', 'double');
         h5create(prefix_clean, '/clean_images/images_imag', [image_size, image_size, total_signals_per_SNR], 'Datatype', 'double');
         h5create(prefix_clean, '/noisy_images/images_real', [image_size, image_size, total_signals_per_SNR], 'Datatype', 'double');
         h5create(prefix_clean, '/noisy_images/images_imag', [image_size, image_size, total_signals_per_SNR], 'Datatype', 'double');
         h5create(prefix_clean, '/labels', [length(waveforms), total_signals_per_SNR]);
     
    end

    parfor snr_index = 1:length(SNR)
        s = RandStream.create('mt19937ar','Seed', seed + snr_index);
        RandStream.setGlobalStream(s);

        prefix_clean = fullfile(output_dir, [prefix resize_method '_' transform '_' num2str(snr_index) '_' 'sigma' '_' num2str(sigma) '.h5']);
        start_index = 1;
        input_batch = complex(zeros(image_size, image_size, signals_per_SNR));
        input_noisy_batch = complex(zeros(image_size, image_size, signals_per_SNR));
        
        for waveform_index = 1:length(waveforms)
            waveform = waveforms{waveform_index};
            output_vector = zeros(1, length(waveforms));
            output_vector(waveform_index) = 1;
            output_data_batch = repmat(output_vector, signals_per_SNR, 1).';

            switch waveform
                case 'LFM'
                    % LFM-specific parameters
                    %disp(['Generating ',waveform, ' waveform on worker ', int2str(method_idx)]);
                    disp(['Generating ',waveform, ' waveform for SNR ', int2str(SNR(snr_index))]);
                    fc = linspace(fs/6,fs/4, signals_per_SNR);
                    fc=fc(randperm(signals_per_SNR)); %random permutation
                    B = linspace(fs/20, fs/16, signals_per_SNR);
                    B = B(randperm(signals_per_SNR)); %random permutation
                    N = linspace(512,1024,signals_per_SNR);
                    N=round(N(randperm(signals_per_SNR))); %random permutation
                    sweepDirections = {'Up','Down'};
    
                    for idx = 1:signals_per_SNR
                        wav = type_LFM(N(idx),fs,A,fc(idx),B(idx),sweepDirections{randi(2)});
                        [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                        resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std, resize_method, transform, sigma);
                        input_batch(:, :,idx) = resized_images.transform_resized;
                        input_noisy_batch(:, :,idx) = resized_images.transform_noisy_resized;
                    end
                    write_batch_to_h5(prefix_clean, input_batch, input_noisy_batch, output_data_batch, start_index, image_size, signals_per_SNR)
                    start_index = start_index + signals_per_SNR;
    
                case 'Costas'
                    %disp(['Generating ',waveform, ' waveform on worker ', int2str(method_idx)]);
                    disp(['Generating ',waveform, ' waveform for SNR ', int2str(SNR(snr_index))]);
                    Lc = [3, 4, 5, 6];
                    fcmin = linspace(fs/24,fs/20,signals_per_SNR);
                    fcmin=fcmin(randperm(signals_per_SNR));
                    N = linspace(512,1024,signals_per_SNR);
                    N=round(N(randperm(signals_per_SNR)));
                    for idx = 1:signals_per_SNR
                        NumHop = randperm(Lc(randi(4)));
                        wav = type_Costas(N(idx), fs, A, fcmin(idx), NumHop);
                        [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                        resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std, resize_method, transform, sigma);
                        input_batch(:, :,idx) = resized_images.transform_resized;
                        input_noisy_batch(:, :,idx) = resized_images.transform_noisy_resized;
                    end
                    write_batch_to_h5(prefix_clean, input_batch, input_noisy_batch, output_data_batch, start_index, image_size, signals_per_SNR)
                    start_index = start_index + signals_per_SNR;
                case 'Barker'
                    %disp(['Generating ',waveform, ' waveform on worker ', int2str(method_idx)]);
                    disp(['Generating ',waveform, ' waveform for SNR ', int2str(SNR(snr_index))]);
                    Lc = [7,11,13];
                    fc = linspace(fs/6,fs/4,signals_per_SNR);
                    fc=fc(randperm(signals_per_SNR));
                    Ncc = 10:13;
                    for idx = 1:signals_per_SNR
                        Bar = Lc(randi(3));
                        if Bar == 7
                            phaseCode = [0 0 0 1 1 0 1]*pi;
                        elseif Bar == 11
                            phaseCode = [0 0 0 1 1 1 0 1 1 0 1]*pi;
                        elseif Bar == 13
                            phaseCode = [0 0 0 0 0 1 1 0 0 1 0 1 0]*pi;
                        end
                        wav = type_Barker(Ncc(randi(length(Ncc))), fs, A, fc(idx), phaseCode);
                        [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                        resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std, resize_method, transform, sigma);
                        input_batch(:, :,idx) = resized_images.transform_resized;
                        input_noisy_batch(:, :,idx) = resized_images.transform_noisy_resized;
                    end
                    write_batch_to_h5(prefix_clean, input_batch, input_noisy_batch, output_data_batch, start_index, image_size, signals_per_SNR)
                    start_index = start_index + signals_per_SNR;
                case 'Frank'
                    %disp(['Generating ',waveform, ' waveform on worker ', int2str(method_idx)]);
                    disp(['Generating ',waveform, ' waveform for SNR ', int2str(SNR(snr_index))]);
                    fc = linspace(fs/4,fs/3,signals_per_SNR);
                    fc=fc(randperm(signals_per_SNR));
                    Ncc = [3,4,5];
                    M = [6, 7, 8];
                    for idx = 1:signals_per_SNR
                        wav = type_Frank(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                        [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                        resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std, resize_method, transform, sigma);
                        input_batch(:, :,idx) = resized_images.transform_resized;
                        input_noisy_batch(:, :,idx) = resized_images.transform_noisy_resized;
                    end
                    write_batch_to_h5(prefix_clean, input_batch, input_noisy_batch, output_data_batch, start_index, image_size, signals_per_SNR)
                    start_index = start_index + signals_per_SNR;
                case 'P1'
                    %disp(['Generating ',waveform, ' waveform on worker ', int2str(method_idx)]);
                    disp(['Generating ',waveform, ' waveform for SNR ', int2str(SNR(snr_index))]);
                    fc = linspace(fs/4,fs/3,signals_per_SNR);
                    fc=fc(randperm(signals_per_SNR));
                    Ncc = [3,4,5];
                    M = [6, 7, 8];
                    for idx = 1:signals_per_SNR
                        wav = type_P1(Ncc(randi(3)), fs, A, fc(idx), M(randi(3)));
                        [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                        resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std, resize_method, transform, sigma);
                        input_batch(:, :,idx) = resized_images.transform_resized;
                        input_noisy_batch(:, :,idx) = resized_images.transform_noisy_resized;
                    end
                    write_batch_to_h5(prefix_clean, input_batch, input_noisy_batch, output_data_batch, start_index, image_size, signals_per_SNR)
                    start_index = start_index + signals_per_SNR;
                case 'P2'
                    %disp(['Generating ',waveform, ' waveform on worker ', int2str(method_idx)]);
                    disp(['Generating ',waveform, ' waveform for SNR ', int2str(SNR(snr_index))]);
                    fc = linspace(fs/4,fs/3,signals_per_SNR);
                    fc=fc(randperm(signals_per_SNR));
                    Ncc = [3,4,5];
                    M = [6, 8];
                    for idx = 1:signals_per_SNR
                        wav = type_P2(Ncc(randi(3)), fs, A, fc(idx), M(randi(2)));
                        [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                        resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std, resize_method, transform, sigma);
                        input_batch(:, :,idx) = resized_images.transform_resized;
                        input_noisy_batch(:, :,idx) = resized_images.transform_noisy_resized;
                    end            
                    write_batch_to_h5(prefix_clean, input_batch, input_noisy_batch, output_data_batch, start_index, image_size, signals_per_SNR)
                    start_index = start_index + signals_per_SNR;
                case 'P3'
                    %disp(['Generating ',waveform, ' waveform on worker ', int2str(method_idx)]);
                    disp(['Generating ',waveform, ' waveform for SNR ', int2str(SNR(snr_index))]);
                    fc = linspace(fs/4,fs/3,signals_per_SNR);
                    fc=fc(randperm(signals_per_SNR));
                    Ncc = [3,4,5];
                    p = [36, 49, 64];
                    for idx = 1:signals_per_SNR
                        wav = type_P3(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                        [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                        resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std, resize_method, transform, sigma);
                        input_batch(:, :,idx) = resized_images.transform_resized;
                        input_noisy_batch(:, :,idx) = resized_images.transform_noisy_resized;
                    end
                    write_batch_to_h5(prefix_clean, input_batch, input_noisy_batch, output_data_batch, start_index, image_size, signals_per_SNR)
                    start_index = start_index + signals_per_SNR;
                case 'P4'
                    %disp(['Generating ',waveform, ' waveform on worker ', int2str(method_idx)]);
                    disp(['Generating ',waveform, ' waveform for SNR ', int2str(SNR(snr_index))]);
                    fc = linspace(fs/4,fs/3,signals_per_SNR);
                    fc=fc(randperm(signals_per_SNR));
                    Ncc = [3,4,5];
                    p = [36, 49, 64];
                    for idx = 1:signals_per_SNR
                        wav = type_P4(Ncc(randi(3)), fs, A, fc(idx), p(randi(3)));
                        [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                        resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std, resize_method, transform, sigma);
                        input_batch(:, :,idx) = resized_images.transform_resized;
                        input_noisy_batch(:, :,idx) = resized_images.transform_noisy_resized;
                    end
                    write_batch_to_h5(prefix_clean, input_batch, input_noisy_batch, output_data_batch, start_index, image_size, signals_per_SNR)
                    start_index = start_index + signals_per_SNR;
                case 'T1'
                    disp(['Generating ',waveform, ' waveform for SNR ', int2str(SNR(snr_index))]);
                    fc = linspace(fs/6,fs/5,signals_per_SNR);
                    fc=fc(randperm(signals_per_SNR));
                    Ng = [4,5,6];
                    N = linspace(512,1024,signals_per_SNR);
                    N=round(N(randperm(signals_per_SNR)));
                    Nps = 2;
                    for idx = 1:signals_per_SNR
                        wav = type_T1(fs, A, fc(idx),Nps,Ng(randi(3)));
                        [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                        resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std, resize_method, transform, sigma);
                        input_batch(:, :,idx) = resized_images.transform_resized;
                        input_noisy_batch(:, :,idx) = resized_images.transform_noisy_resized;
                    end
                    write_batch_to_h5(prefix_clean, input_batch, input_noisy_batch, output_data_batch, start_index, image_size, signals_per_SNR)
                    start_index = start_index + signals_per_SNR;
                    
                case 'T2'
                    disp(['Generating ',waveform, ' waveform for SNR ', int2str(SNR(snr_index))]);
                    fc = linspace(fs/6,fs/5,signals_per_SNR);
                    fc=fc(randperm(signals_per_SNR));
                    Ng = [4,5,6];
                    Nps = 2;
                    N = linspace(512,1024,signals_per_SNR);
                    N=round(N(randperm(signals_per_SNR)));
                    for idx = 1:signals_per_SNR
                        wav = type_T2(fs, A, fc(idx),Nps,Ng(randi(3)));
                        [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                        resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std, resize_method, transform, sigma);
                        input_batch(:, :,idx) = resized_images.transform_resized;
                        input_noisy_batch(:, :,idx) = resized_images.transform_noisy_resized;
                    end
                    write_batch_to_h5(prefix_clean, input_batch, input_noisy_batch, output_data_batch, start_index, image_size, signals_per_SNR)
                    start_index = start_index + signals_per_SNR;
                    
                case 'T3'
                    disp(['Generating ',waveform, ' waveform for SNR ', int2str(SNR(snr_index))]);
                    fc = linspace(fs/6,fs/5,signals_per_SNR);
                    fc=fc(randperm(signals_per_SNR));
                    B = linspace(fs/20,fs/10,signals_per_SNR);
                    B = B(randperm(signals_per_SNR));
                    Ng = [4,5,6];
                    N = linspace(512,1024,signals_per_SNR);
                    N=round(N(randperm(signals_per_SNR)));
                    for idx = 1:signals_per_SNR
                        wav = type_T3(N(idx), fs, A, fc(idx), Nps,B(idx));
                        [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                        resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std, resize_method, transform, sigma);
                        input_batch(:, :,idx) = resized_images.transform_resized;
                        input_noisy_batch(:, :,idx) = resized_images.transform_noisy_resized;
                    end
                    write_batch_to_h5(prefix_clean, input_batch, input_noisy_batch, output_data_batch, start_index, image_size, signals_per_SNR)
                    start_index = start_index + signals_per_SNR;
                    
                case 'T4'
                    disp(['Generating ',waveform, ' waveform for SNR ', int2str(SNR(snr_index))]);
                    fc = linspace(fs/6,fs/5,signals_per_SNR);
                    fc=fc(randperm(signals_per_SNR));
                    B = linspace(fs/20,fs/10,signals_per_SNR);
                    B = B(randperm(signals_per_SNR));
                    Ng = [4,5,6];
                    N = linspace(512,1024,signals_per_SNR);
                    N=round(N(randperm(signals_per_SNR)));
                    for idx = 1:signals_per_SNR
                        wav = type_T4(N(idx), fs, A, fc(idx), Nps,B(idx));
                        [noisy_signal, real_noise_std] = merge_noise(wav, SNR(snr_index));
                        resized_images = transform_data(wav, noisy_signal, 1024, image_size, real_noise_std, resize_method, transform, sigma);
                        input_batch(:, :,idx) = resized_images.transform_resized;
                        input_noisy_batch(:, :,idx) = resized_images.transform_noisy_resized;
                    end
                    write_batch_to_h5(prefix_clean, input_batch, input_noisy_batch, output_data_batch, start_index, image_size, signals_per_SNR)
                    start_index = start_index + signals_per_SNR;
            end
       end
    end
    
    delete(pool);
    if train
        disp("Starting data shuffling");
        %combine_h5_files_shuffled(resize_method, transform, train, length(SNR), signals_per_SNR, length(waveforms), image_size, output_dir);
        combine_h5_files(resize_method, transform, train, length(SNR), signals_per_SNR, length(waveforms), image_size, output_dir, sigma);
    else
        combine_h5_files(resize_method, transform, train, length(SNR), signals_per_SNR, length(waveforms), image_size, output_dir, sigma);
    end
end


function write_batch_to_h5(prefix_clean, input_batch, input_noisy_batch, output_data_batch, start, image_size, signals_per_SNR)
    input_batch = permute(input_batch, [2, 1, 3]);
    input_noisy_batch = permute(input_noisy_batch, [2, 1, 3]);
    h5write(prefix_clean, '/clean_images/images_real', real(input_batch), [1 1 start], [image_size, image_size, signals_per_SNR]);
    h5write(prefix_clean, '/clean_images/images_imag', imag(input_batch), [1 1 start], [image_size, image_size, signals_per_SNR]);
    h5write(prefix_clean, '/noisy_images/images_real', real(input_noisy_batch), [1 1 start], [image_size, image_size, signals_per_SNR]);
    h5write(prefix_clean, '/noisy_images/images_imag', imag(input_noisy_batch), [1 1 start], [image_size, image_size, signals_per_SNR]);
    h5write(prefix_clean, '/labels', output_data_batch, [1 start], [size(output_data_batch, 1) signals_per_SNR]);
end

function combine_h5_files(resize_method, transform, train, snr_length, signals_per_SNR, num_waveforms, image_size, output_dir, sigma)

    if train
        prefix = 'input_train_';
    else
        prefix = 'input_test_';
    end

    total_signals = snr_length * signals_per_SNR * num_waveforms;
  
    combined_clean_file = fullfile(output_dir, [prefix resize_method '_' transform '_' 'sigma' '_' num2str(sigma) '.h5']);

    h5create(combined_clean_file, '/clean_images/images_real', [image_size, image_size, total_signals], 'Datatype', 'double');
    h5create(combined_clean_file, '/clean_images/images_imag', [image_size, image_size, total_signals], 'Datatype', 'double');
    h5create(combined_clean_file, '/noisy_images/images_real', [image_size, image_size, total_signals], 'Datatype', 'double');
    h5create(combined_clean_file, '/noisy_images/images_imag', [image_size, image_size, total_signals], 'Datatype', 'double');
    h5create(combined_clean_file, '/labels', [num_waveforms, snr_length * signals_per_SNR * num_waveforms])

    current_index = 1;
    for snr_index = 1:snr_length
        prefix_clean = fullfile(output_dir, [prefix resize_method '_' transform '_' num2str(snr_index) '_' 'sigma' '_' num2str(sigma) '.h5']);

        for waveform_index = 1:num_waveforms
            num_signals_to_read = signals_per_SNR;
            
            batch_start_index = (waveform_index - 1) * num_signals_to_read + 1;
            %Read and write clean data
            real_data = h5read(prefix_clean, '/clean_images/images_real', [1,1, batch_start_index], [image_size,image_size, num_signals_to_read]);
            imag_data = h5read(prefix_clean, '/clean_images/images_imag', [1,1, batch_start_index], [image_size,image_size, num_signals_to_read]);
            h5write(combined_clean_file, '/clean_images/images_real', real_data, [1,1,current_index], [image_size,image_size, num_signals_to_read]);
            h5write(combined_clean_file, '/clean_images/images_imag', imag_data, [1,1,current_index], [image_size,image_size, num_signals_to_read]);
            
            %Read and write noisy data
            real_data_noisy = h5read(prefix_clean, '/noisy_images/images_real', [1,1, batch_start_index], [image_size,image_size, num_signals_to_read]);
            imag_data_noisy = h5read(prefix_clean, '/noisy_images/images_imag', [1,1, batch_start_index], [image_size,image_size, num_signals_to_read]);
            h5write(combined_clean_file, '/noisy_images/images_real', real_data_noisy, [1,1,current_index], [image_size,image_size, num_signals_to_read]);
            h5write(combined_clean_file, '/noisy_images/images_imag', imag_data_noisy, [1,1,current_index], [image_size,image_size, num_signals_to_read]);
            
            output_data = h5read(prefix_clean, '/labels', [1, batch_start_index], [num_waveforms, num_signals_to_read]);
            h5write(combined_clean_file, '/labels', output_data, [1, current_index], [num_waveforms, num_signals_to_read]);
            current_index = current_index + num_signals_to_read;
        end
        delete(prefix_clean);
    end
end

function combine_h5_files_shuffled(resize_method, transform, train, snr_length, signals_per_SNR, num_waveforms, image_size, output_dir, sigma)
    if train
        prefix = 'input_train_';
    else
        prefix = 'input_test_';
    end
    total_signals = snr_length * signals_per_SNR * num_waveforms;
    
    combined_file = fullfile(output_dir, [prefix resize_method '_' transform '_' 'sigma' '_' num2str(sigma) '.h5']);
    h5create(combined_file, '/clean_images/images_real', [image_size, image_size, total_signals], 'Datatype', 'double');
    h5create(combined_file, '/clean_images/images_imag', [image_size, image_size, total_signals], 'Datatype', 'double');
    h5create(combined_file, '/noisy_images/images_real', [image_size, image_size, total_signals], 'Datatype', 'double');
    h5create(combined_file, '/noisy_images/images_imag', [image_size, image_size, total_signals], 'Datatype', 'double');
    h5create(combined_file, '/labels', [total_signals, num_waveforms], 'Datatype', 'double');

    rng('shuffle'); 
    % Create a single permutation for ALL signals
    permutation = randperm(total_signals);
    
    for snr_index = 1:snr_length
        prefix_clean = fullfile(output_dir, [prefix resize_method '_' transform '_' num2str(snr_index) '_' 'sigma' '_' num2str(sigma) '.h5']);
        disp(['Running on SNR index ' int2str(snr_index)]);
        for waveform_index = 1:num_waveforms
            num_signals_to_read = signals_per_SNR;

            % Calculate the start and end indices in the combined dataset for current waveform and snr
            start_in_combined = (snr_index - 1) * signals_per_SNR * num_waveforms + (waveform_index - 1) * signals_per_SNR + 1;
            end_in_combined = start_in_combined + num_signals_to_read - 1;

            indices_in_combined = start_in_combined:end_in_combined;

            % Get the shuffled indices from the overall permutation
            shuffled_indices = permutation(indices_in_combined);

            % Calculate indices to read from the original h5 file
            batch_start_index = (waveform_index - 1) * num_signals_to_read + 1;

            % Read data from the h5 file
            real_data = h5read(prefix_clean, '/clean_images/images_real', [1,1, batch_start_index], [image_size,image_size, num_signals_to_read]);
            imag_data = h5read(prefix_clean, '/clean_images/images_imag', [1,1, batch_start_index], [image_size,image_size, num_signals_to_read]);
            noisy_real_data = h5read(prefix_clean, '/noisy_images/images_real', [1,1, batch_start_index], [image_size,image_size, num_signals_to_read]);
            noisy_imag_data = h5read(prefix_clean, '/noisy_images/images_imag', [1,1, batch_start_index], [image_size,image_size, num_signals_to_read]);
            output_data = h5read(prefix_clean, '/labels', [1, batch_start_index], [num_waveforms, num_signals_to_read]);
            
            % Write data to the combined file at the shuffled positions
            for i = 1:num_signals_to_read
                combined_index = shuffled_indices(i);
                h5write(combined_file, '/clean_images/images_real', real_data(i,:,:), [1,1, combined_index], [image_size,image_size, 1]);
                h5write(combined_file, '/clean_images/images_imag', imag_data(i,:,:), [1,1, combined_index], [image_size,image_size, 1]);
                h5write(combined_file, '/noisy_images/images_real', noisy_real_data(i,:,:), [1,1, combined_index], [image_size,image_size, 1]);
                h5write(combined_file, '/noisy_images/images_imag', noisy_imag_data(i,:,:), [1,1, combined_index], [image_size,image_size, 1]);
                h5write(combined_file, '/labels', output_data(i,:), [1, combined_index], [num_waveforms, 1]);
            end
        end
        delete(prefix_clean);
    end
end