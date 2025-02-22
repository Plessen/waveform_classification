import matplotlib.pyplot as plt
import pandas as pd
    
def plot_all_acc(df_list, number_waveforms, SNR, waveform_list, label_list):
    
    rows = number_waveforms // 2 if number_waveforms % 2 == 0 else number_waveforms // 2 + 1
    
    figure, axis = plt.subplots(rows, 2)
    
    for j, df in enumerate(df_list):
        for i in range(number_waveforms):
            test_acc = df["test_acc_step"]
            test_acc = test_acc[test_acc.notna()]
            test_acc_waveform = test_acc.iloc[i::number_waveforms]

            axis[i//2, i%2].plot(SNR, test_acc_waveform, label=label_list[j])
            axis[i//2, i%2].set_title("{}".format(waveform_list[i]))
            axis[i//2, i%2].legend()
    
    plt.show()

def get_dataframe(location_folder, model_name, version):
    csv_path = "{}/{}/{}/metrics.csv".format(location_folder, model_name, version)
    df = pd.read_csv(csv_path)
    return df

def plot_overall_test_accuracy(df_list, number_waveforms, SNR, label_list):
    
    for i, df in enumerate(df_list):
        test_acc = df["test_acc_step"]
        test_acc = test_acc[test_acc.notna()]
        
        test_overall_acc = test_acc.values.reshape(-1, number_waveforms).mean(axis=1)
        plt.plot(SNR, test_overall_acc, label=label_list[i])
        plt.legend()

    plt.show()
    
df = get_dataframe("logs_cluster", "realcnn-attention-nearest-SST-0.05", "version_0")
df1 = get_dataframe("logs_cluster","realcnn-attention-nearest-SST-0.06", "version_0")
df2 = get_dataframe("logs_cluster","realcnn-attention-nearest-SST-0.07", "version_0")
df3 = get_dataframe("logs_cluster","realcnn-attention-nearest-SST-0.08", "version_0")
df4 = get_dataframe("logs_cluster","realcnn-attention-nearest-SST-0.09", "version_0")
df5 = get_dataframe("logs_cluster","realcnn-attention-nearest-SST-0.1", "version_0")

df_list = [df, df1, df2, df3, df4, df5]
number_waveforms = 8
SNR = [i for i in range(-14, -2, 2)]
waveform_list = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4']
label_list = ["Sigma - 0.05", "Sigma - 0.06", "Sigma - 0.07", "Sigma - 0.08", "Sigma - 0.09", "Sigma - 0.1"]
plot_overall_test_accuracy(df_list, number_waveforms, SNR, label_list)
plot_all_acc(df_list, number_waveforms, SNR, waveform_list, label_list)


df = get_dataframe("logs_cluster", "realcnn-attention-nearest-SST-sigma-0.04-all-waveforms", "version_0")
df1 = get_dataframe("logs_cluster","realcnn-attention-nearest-SST-sigma-0.05-all-waveforms", "version_0")
df2 = get_dataframe("logs_cluster","realcnn-attention-nearest-SST-sigma-0.06-all-waveforms", "version_0")

df3 = get_dataframe("logs_cluster","realcnn-attention-bilinear-SST-sigma-0.05-all-waveforms", "version_0")
df4 = get_dataframe("logs_cluster","realcnn-attention-bicubic-SST-sigma-0.05-all-waveforms", "version_0")
df5 = get_dataframe("logs_cluster","realcnn-attention-lanczos2-SST-sigma-0.05-all-waveforms", "version_0")
df5 = get_dataframe("logs_cluster","realcnn-attention-lanczos3-SST-sigma-0.05-all-waveforms", "version_0")

df_list = [df, df1, df2, df3, df4, df5]
number_waveforms = 12
SNR = [i for i in range(-14, -2, 2)]
waveform_list = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4', 'T1', 'T2', 'T3', 'T4']
label_list = ["Sigma - 0.04", "Sigma - 0.05", "Sigma - 0.06", "Bilinear", "Bicubic", "Lanczos2", "Lanczos3"]
plot_overall_test_accuracy(df_list, number_waveforms, SNR, label_list)
plot_all_acc(df_list, number_waveforms, SNR, waveform_list, label_list)




df6 = get_dataframe("logs_cluster", "realcnn-attention-nearest-SST", "version_1")
df7 = get_dataframe("logs_cluster","realcnn-attention-bilinear-SST", "version_1")
df8 = get_dataframe("logs_cluster","realcnn-attention-bicubic-SST", "version_1")
df9 = get_dataframe("logs_cluster","realcnn-attention-lanczos2-SST", "version_1")
df10 = get_dataframe("logs_cluster","realcnn-attention-lanczos3-SST", "version_1")

df_list = [df6, df7, df8, df9, df10]
number_waveforms = 8
SNR = [i for i in range(-14, -2, 2)]
waveform_list = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4']
label_list = ["nearest", "bilinear", "bicubic", "lanczos2", "lanczos3"]
plot_overall_test_accuracy(df_list, number_waveforms, SNR, label_list)
plot_all_acc(df_list, number_waveforms, SNR, waveform_list, label_list)
