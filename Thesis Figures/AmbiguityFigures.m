fs = 1000;
samples = 1000;
A = 1;
output_path = 'C:\Users\serge\Documents\Statistical Signal Processing\Thesis\FSSTn-master\FSSTn-master\Thesis Figures';

function save_ambiguity(signal, fs, output_path, output_file_name, order, doppler_normalization, contour_level)
    % Calculate the signal duration
    T = (length(signal) / fs);
    
    % Compute the ambiguity function
    [ambiguity, delay, doppler] = ambgfun(signal, fs, 1 / T);
    delay_normalized = delay / (T / order);
    delay_text = '\tau / T';
    if order > 1
        delay_text = '\tau / t_b';
    end
    if doppler_normalization == 1
        doppler_normalized = doppler * (T / order);
        doppler_text = '\nu \cdot t_b';
    else
        doppler_normalized = doppler * T;
        doppler_text = '\nu \cdot T';
    end
    if order * T >= 12 || order == 1
        positive_doppler_indices = doppler_normalized >= 0 & doppler_normalized <= 12;
    else
        positive_doppler_indices = doppler_normalized >= 0 & doppler_normalized <= T * order;
    end
    %% Save 3D Mesh Plot
    figure('units','normalized','outerposition',[0 0 1 1]);
    mesh(delay_normalized, doppler_normalized(positive_doppler_indices), ...
        abs(ambiguity(positive_doppler_indices, :)));
    xlabel_handle = xlabel(delay_text, 'FontSize', 40, "Rotation", 20);
    ylabel_handle = ylabel(doppler_text, 'FontSize', 40, "Rotation", -30);
    zlabel('Magnitude', 'FontSize', 40);
    set(gca, 'FontSize', 30);

    %axis tight;

    xlimits = get(gca, 'XLim');
    ylimits = get(gca, 'YLim');
    zlimits = get(gca, 'ZLim');
    
    set(xlabel_handle, 'Position', [(xlimits(1) + xlimits(2)) / 2, ylimits(1) - 0.2 * range(ylimits), zlimits(1)]);
    set(ylabel_handle, 'Position', [xlimits(1) - 0.2 * range(xlimits), (ylimits(1) + ylimits(2)) / 2, zlimits(1)]);
    % Save the 3D plot as EPS
    output_filename = fullfile(output_path, output_file_name); % Specify desired file name
    print('-depsc', output_filename);
    close; % Close the figure after saving
    
    %% Save 2D imagesc Plot
    figure('units','normalized','outerposition',[0 0 1 1]);

    if contour_level == 1
    doppler_indicies = doppler_normalized >= -T * order & doppler_normalized <= T * order;
    contour(delay_normalized, doppler_normalized(doppler_indicies), ...
        abs(ambiguity(doppler_indicies, :)), 1 / (order + 1): 0.001 : 1 / order + 0.001);
    axis xy; % Ensure delay and Doppler axes are in the correct orientation
    xlabel(delay_text, 'FontSize', 40);
    ylabel(doppler_text, 'FontSize', 40);
    colorbar;
    set(gca, 'FontSize', 30);
    else
    contour(delay_normalized, doppler_normalized(positive_doppler_indices), ...
        abs(ambiguity(positive_doppler_indices, :)));
    axis xy; % Ensure delay and Doppler axes are in the correct orientation
    xlabel(delay_text, 'FontSize', 40);
    ylabel(doppler_text, 'FontSize', 40);
    colorbar;
    set(gca, 'FontSize', 30);
    end
    % Save the imagesc plot as EPS with "_imagesc" suffix
    [~, name, ext] = fileparts(output_file_name); % Extract name and extension
    imagesc_filename = fullfile(output_path, [name '_contour' ext]);
    print('-depsc', imagesc_filename);
    close; % Close the figure after saving
end
%Freq mod
lfm_signal = type_LFM(samples,fs,A, fs/6, 10,"UP");
costas_signal = type_Costas(samples, fs, A, samples / fs * 10, [1 3 2 6 4 5]);

%Phase
phaseCode = [0 0 0 0 0 1 1 0 0 1 0 1 0]*pi;
barker_signal = type_Barker(20, fs, A, fs/5, phaseCode);

%PolyPhase
frank_signal = type_Frank(5, fs, A, fs / 4, 8);
p1_signal = type_P1(5, fs, A, fs / 4, 8);
p2_signal = type_P2(5, fs, A, fs / 4, 8);
p3_signal = type_P3(5, fs, A, fs / 4, 64);
p4_signal = type_P4(5, fs, A, fs  / 4, 64);

%PolyTime
t1_signal = type_T1(fs, A, fs / 5,2, 6);
t2_signal = type_T2(fs, A, fs / 5, 2, 6);
t3_signal = type_T3(samples, fs, A, fs / 5, 2, fs / 10);
t4_signal = type_T4(samples, fs, A, fs / 5, 2, fs / 10);

% LFM Signal
save_ambiguity(lfm_signal, fs, output_path, 'lfm_signal_ambiguity.eps', 1, 0, 0);
save_ambiguity(costas_signal, fs, output_path, 'costas_signal_ambiguity.eps', 6, 1, 1);
save_ambiguity(barker_signal, fs, output_path, 'barker_signal_ambiguity.eps', 13, 0, 0);
save_ambiguity(frank_signal, fs, output_path, 'frank_signal_ambiguity.eps', 8, 0, 0);
save_ambiguity(p1_signal, fs, output_path, 'p1_signal_ambiguity.eps', 8, 0, 0);
save_ambiguity(p2_signal, fs, output_path, 'p2_signal_ambiguity.eps', 8, 0, 0);
save_ambiguity(p3_signal, fs, output_path, 'p3_signal_ambiguity.eps', 64, 0, 0);
save_ambiguity(p4_signal, fs, output_path, 'p4_signal_ambiguity.eps', 64, 0, 0);
save_ambiguity(t1_signal, fs, output_path, 't1_signal_ambiguity.eps', 6, 0, 0);
save_ambiguity(t2_signal, fs, output_path, 't2_signal_ambiguity.eps', 6, 0, 0);
save_ambiguity(t3_signal, fs, output_path, 't3_signal_ambiguity.eps', 1, 0, 0);
save_ambiguity(t4_signal, fs, output_path, 't4_signal_ambiguity.eps', 1, 0, 0);
