% Define the function to read CSV files into tables
function df = get_dataframe(location_folder, model_name, version)
    csv_path = fullfile(location_folder, model_name, version, 'metrics.csv');
    df = readtable(csv_path);
end

% Define the function to plot overall test accuracy
function plot_overall_test_accuracy(df_list, number_waveforms, SNR, label_list, ylim_interval)
    %figure("Position",[10 10 900 600]);
    figure("Position",[10 10 900 600]);
    hold on;
    ax = gca;
    ax.FontSize = 14; 
    xlabel("SNR", "FontSize", 22);
    ylabel("Accuracy","FontSize", 22);
    markers = {'o', 's', 'd', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+'};
    linestyles = {'-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':'};
    for i = 1:length(df_list)
        df = df_list{i};
        test_acc = df.test_acc_step;
        test_acc = test_acc(~isnan(test_acc));
        test_overall_acc = mean(reshape(test_acc, number_waveforms, []), 1);
        plot(SNR, test_overall_acc, 'DisplayName', label_list{i}, ...
            'LineWidth', 2, 'Marker', markers{i}, 'LineStyle', linestyles{i});
    end
    legend;
    index = -14 <= SNR & SNR <= -12;
    axes('Position', [0.4 0.2 0.4 0.4]);
    box on;
    hold on;
    for i = 1:length(df_list)
        df = df_list{i};
        test_acc = df.test_acc_step;
        test_acc = test_acc(~isnan(test_acc));
        test_overall_acc = mean(reshape(test_acc, number_waveforms, []), 1);
        plot(SNR(index), test_overall_acc(index), 'DisplayName', label_list{i}, ...
            'LineWidth', 2, 'Marker', markers{i}, 'LineStyle', linestyles{i});
    end
    ax = gca;
    ax.FontSize = 13; 
    ylim(ylim_interval);
    %ylim([0.881 0.905]);
    hold off;
end

% Define the function to plot accuracy for each waveform
function plot_all_acc(df_list, number_waveforms, SNR, waveform_list, label_list)
    markers = {'o', 's', 'd', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+'};
    linestyles = {'-', '--', '-.', ':', '-', '--', '-.', ':', '-', '--', '-.', ':'};
    figure('Position', [10 10 1800 600]);
    for i = 1:number_waveforms

        need_subplot = false;

        %if mod(i-1, 6) == 0
        %    figure('Position', [10 10 1200 600]);
        %end
        %figure('Position', [10 10 900 900])
        subplot_handle = subplot(2, 6, mod(i-1, 12) + 1);
        hold on;
        xlabel("SNR", "FontSize", 15);
        ylabel("Accuracy","FontSize", 15);
        xlim([-14 -4]);
        for j = 1:length(df_list)
            df = df_list{j};
            test_acc = df.test_acc_step;
            test_acc = test_acc(~isnan(test_acc));
            test_acc_waveform = test_acc(i:number_waveforms:end);

            if(sum(test_acc_waveform <= 0.9) > 0)
                need_subplot = true;
            end

            plot(SNR, test_acc_waveform, 'DisplayName', label_list{j}, ...
                'LineWidth', 2, 'Marker', markers{j}, 'LineStyle', linestyles{j});
        end
        title(waveform_list{i});
        legend;
        hold off;

        if need_subplot == true
            disp(waveform_list{i})
            index = -14 <= SNR & SNR <= -12;
            axes('Position', get(subplot_handle, 'Position').*[1 1 0.4 0.4] + [0.05 0.02 0 0]);
            box on;
            hold on;
            for j = 1:length(df_list)
                df = df_list{j};
                test_acc = df.test_acc_step;
                test_acc = test_acc(~isnan(test_acc));
                test_acc_waveform = test_acc(i:number_waveforms:end);

                plot(SNR(index), test_acc_waveform(index), 'DisplayName', label_list{j}, ...
                    'LineWidth', 2, 'Marker', markers{j}, 'LineStyle', linestyles{j});
            end
            %ylim([0.75 0.9]);
            xlim([-14, -13.5]);
        end
        
        hold off;
    end
end

% Main script
%df = get_dataframe('logs_cluster', 'realcnn-attention-nearest-SST-sigma-0.03-all-waveforms', 'version_0');
df1 = get_dataframe('logs_cluster', 'realcnn-attention-nearest-SST-sigma-0.04-all-waveforms', 'version_0');
df2 = get_dataframe('logs_cluster', 'realcnn-attention-nearest-SST-sigma-0.05-all-waveforms', 'version_0');
df3 = get_dataframe('logs_cluster', 'realcnn-attention-nearest-SST-sigma-0.06-all-waveforms', 'version_0');
df4 = get_dataframe('logs', 'realcnn-attention-nearest-SST-sigma-0.07-all-waveforms', 'version_0');
df5 = get_dataframe('logs', 'realcnn-attention-nearest-SST-sigma-0.08-all-waveforms', 'version_0');
%df6 = get_dataframe('logs_cluster', 'realcnn-attention-bilinear-VSST-0.04-all-waveforms', 'version_0');



%df3 = get_dataframe('logs_cluster', 'realcnn-attention-nearest-CWD-sigma-1-all-waveforms', 'version_0');
df_list = {df1, df2, df3, df4, df5};
number_waveforms = 12;
SNR = -14:2:-4;
waveform_list = {'LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4'};
label_list = {'\sigma_g = 0.04', '\sigma_g = 0.05', '\sigma_g = 0.06', '\sigma_g = 0.07', '\sigma_g = 0.08', 'Lanczos3'};

plot_overall_test_accuracy(df_list, number_waveforms, SNR, label_list, [0.881 0.905]);
plot_all_acc(df_list, number_waveforms, SNR, waveform_list, label_list);

df7 = get_dataframe('logs_cluster', 'realcnn-attention-nearest-SST-sigma-0.04-all-waveforms', 'version_0');
df8 = get_dataframe('logs_cluster', 'realcnn-attention-bilinear-SST-sigma-0.04-all-waveforms', 'version_0');
df9 = get_dataframe('logs_cluster', 'realcnn-attention-bicubic-SST-sigma-0.04-all-waveforms', 'version_0');
df10 = get_dataframe('logs_cluster', 'realcnn-attention-lanczos2-SST-sigma-0.04-all-waveforms', 'version_0');
df11 = get_dataframe('logs_cluster', 'realcnn-attention-lanczos3-SST-sigma-0.04-all-waveforms', 'version_0');

df_list = {df7, df8, df9, df10, df11};
number_waveforms = 12;
SNR = -14:2:-4;
waveform_list = {'LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4'};
label_list = {'nearest', 'bilinear', 'bicubic', 'lanczos2', 'lanczos3'};

plot_overall_test_accuracy(df_list, number_waveforms, SNR, label_list, [0.885 0.905]);
%plot_all_acc(df_list, number_waveforms, SNR, waveform_list, label_list);