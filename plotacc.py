import pandas as pd
import matplotlib.pyplot as plt

def plot_test_accuracy_waveform(df, number_waveforms, waveform, SNR):
    test_acc = df["test_acc_step"]
    test_acc = test_acc[test_acc.notna()]
    test_acc_waveform = test_acc.iloc[waveform::number_waveforms]
    
    plt.plot(SNR, test_acc_waveform, label='Test Accuracy')
    plt.show()

def plot_overall_test_accuracy(df, number_waveforms, SNR):
    test_acc = df["test_acc_step"]
    test_acc = test_acc[test_acc.notna()]
    
    test_overall_acc = test_acc.values.reshape(-1, number_waveforms).mean(axis=1)
    plt.plot(SNR, test_overall_acc, label='Test Accuracy')
    

number_waveforms = 8
#model_name = "real_mode_cnn_sst_nearest"
#csv_path = "logs/{}/version_0/metrics.csv".format(model_name)
#df = pd.read_csv(csv_path)

model_name = "complex_attention_model_cnn_sst_nearest"
csv_path = "logs/{}/version_1/metrics.csv".format(model_name)
df2 = pd.read_csv(csv_path)

#plot_overall_test_accuracy(df, number_waveforms,  [i for i in range(-14, -2, 2)])
plot_overall_test_accuracy(df2, number_waveforms,  [i for i in range(-14, -2, 2)])
plt.show()