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
    csv_path = "{}/{}/version_0/metrics.csv".format(location_folder, model_name, version)
    df = pd.read_csv(csv_path)
    return df

df6 = get_dataframe("logs_cluster", "realcnn-attention-bilinear-SST", "version_1")
df7 = get_dataframe("logs_cluster","realcnn-attention-nearest-SST", "version_1")
df8 = get_dataframe("logs_cluster","realcnn-attention-lanczos2-SST", "version_1")
df9 = get_dataframe("logs_cluster","realcnn-attention-lanczos3-SST", "version_1")
df10 = get_dataframe("logs_cluster","realcnn-attention-bicubic-SST", "version_1")

df_list = [df6, df7, df8, df9, df10]
number_waveforms = 8
SNR = [i for i in range(-14, -2, 2)]
waveform_list = ['LFM', 'Costas', 'Barker', 'Frank', 'P1', 'P2', 'P3', 'P4']
label_list = ["Bilinear", "Nearest", "Lanczos2", "Lanczos3", "Bicubic"]

plot_all_acc(df_list, number_waveforms, SNR, waveform_list, label_list)

