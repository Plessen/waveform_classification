import pandas as pd
import matplotlib.pyplot as plt

def plot_test_accuracy_waveform(df, number_waveforms, waveform, SNR):
    test_acc = df["test_acc_step"]
    test_acc = test_acc[test_acc.notna()]
    test_acc_waveform = test_acc.iloc[waveform::number_waveforms]
    
    plt.plot(SNR, test_acc_waveform, label='Test Accuracy')
    plt.show()

def plot_overall_test_accuracy(df, number_waveforms, SNR, label):
    test_acc = df["test_acc_step"]
    test_acc = test_acc[test_acc.notna()]
    
    test_overall_acc = test_acc.values.reshape(-1, number_waveforms).mean(axis=1)
    plt.plot(SNR, test_overall_acc, label=label)
    plt.legend()
    
number_waveforms = 8
model_name = "realcnn-attention-bilinear-VSST"
csv_path = "logs_cluster/{}/version_0/metrics.csv".format(model_name)
df = pd.read_csv(csv_path)

model_name = "realcnn-attention-nearest-VSST"
csv_path = "logs_cluster/{}/version_0/metrics.csv".format(model_name)
df2 = pd.read_csv(csv_path)

model_name = "realcnn-attention-lanczos2-VSST"
csv_path = "logs_cluster/{}/version_0/metrics.csv".format(model_name)
df3 = pd.read_csv(csv_path)

model_name = "realcnn-attention-lanczos3-VSST"
csv_path = "logs_cluster/{}/version_0/metrics.csv".format(model_name)
df4 = pd.read_csv(csv_path)

model_name = "realcnn-attention-bicubic-VSST"
csv_path = "logs_cluster/{}/version_0/metrics.csv".format(model_name)
df5 = pd.read_csv(csv_path)


model_name = "realcnn-attention-bilinear-SST"
csv_path = "logs_cluster/{}/version_0/metrics.csv".format(model_name)
df6 = pd.read_csv(csv_path)

model_name = "realcnn-attention-nearest-SST"
csv_path = "logs_cluster/{}/version_0/metrics.csv".format(model_name)
df7 = pd.read_csv(csv_path)

model_name = "realcnn-attention-lanczos2-SST"
csv_path = "logs_cluster/{}/version_0/metrics.csv".format(model_name)
df8 = pd.read_csv(csv_path)

model_name = "realcnn-attention-lanczos3-SST"
csv_path = "logs_cluster/{}/version_0/metrics.csv".format(model_name)
df9 = pd.read_csv(csv_path)

model_name = "realcnn-attention-bicubic-SST"
csv_path = "logs_cluster/{}/version_0/metrics.csv".format(model_name)
df10 = pd.read_csv(csv_path)

plot_overall_test_accuracy(df, number_waveforms,  [i for i in range(-14, -2, 2)], "Bilinear-VSST-monitor-loss")
plot_overall_test_accuracy(df2, number_waveforms,  [i for i in range(-14, -2, 2)], "Nearest-VSST-monitor-loss")
plot_overall_test_accuracy(df3, number_waveforms,  [i for i in range(-14, -2, 2)], "Lanczos2-VSST-monitor-loss")
#plot_overall_test_accuracy(df4, number_waveforms,  [i for i in range(-14, -2, 2)], "Lanczos3")
plot_overall_test_accuracy(df5, number_waveforms,  [i for i in range(-14, -2, 2)], "Bicubic-VSST-monitor-loss")

plot_overall_test_accuracy(df6, number_waveforms,  [i for i in range(-14, -2, 2)], "Bilinear-SST-monitor-loss")
#plot_overall_test_accuracy(df7, number_waveforms,  [i for i in range(-14, -2, 2)], "Nearest")
plot_overall_test_accuracy(df8, number_waveforms,  [i for i in range(-14, -2, 2)], "Lanczos2-SST-monitor-loss")
plot_overall_test_accuracy(df9, number_waveforms,  [i for i in range(-14, -2, 2)], "Lanczos3-SST-monitor-loss")
plot_overall_test_accuracy(df10, number_waveforms,  [i for i in range(-14, -2, 2)], "Bicubic-SST-monitor-loss")

model_name = "realcnn-attention-bilinear-SST"
csv_path = "logs_cluster/{}/version_1/metrics.csv".format(model_name)
df6 = pd.read_csv(csv_path)

model_name = "realcnn-attention-nearest-SST"
csv_path = "logs_cluster/{}/version_1/metrics.csv".format(model_name)
df7 = pd.read_csv(csv_path)

model_name = "realcnn-attention-lanczos2-SST"
csv_path = "logs_cluster/{}/version_1/metrics.csv".format(model_name)
df8 = pd.read_csv(csv_path)

model_name = "realcnn-attention-lanczos3-SST"
csv_path = "logs_cluster/{}/version_1/metrics.csv".format(model_name)
df9 = pd.read_csv(csv_path)

model_name = "realcnn-attention-bicubic-SST"
csv_path = "logs_cluster/{}/version_1/metrics.csv".format(model_name)
df10 = pd.read_csv(csv_path)

plot_overall_test_accuracy(df6, number_waveforms,  [i for i in range(-14, -2, 2)], "Bilinear-SST-monitor-val")
plot_overall_test_accuracy(df7, number_waveforms,  [i for i in range(-14, -2, 2)], "Nearest-monitor-val")
plot_overall_test_accuracy(df8, number_waveforms,  [i for i in range(-14, -2, 2)], "Lanczos2-SST-monitor-val")
plot_overall_test_accuracy(df9, number_waveforms,  [i for i in range(-14, -2, 2)], "Lanczos3-SST-monitor-val")
plot_overall_test_accuracy(df10, number_waveforms,  [i for i in range(-14, -2, 2)], "Bicubic-SST-monitor-val")


model_name = "realcnn-attention-bilinear-VSST"
csv_path = "logs_cluster/{}/version_1/metrics.csv".format(model_name)
df6 = pd.read_csv(csv_path)

model_name = "realcnn-attention-nearest-VSST"
csv_path = "logs_cluster/{}/version_1/metrics.csv".format(model_name)
df7 = pd.read_csv(csv_path)

model_name = "realcnn-attention-lanczos2-VSST"
csv_path = "logs_cluster/{}/version_1/metrics.csv".format(model_name)
df8 = pd.read_csv(csv_path)

model_name = "realcnn-attention-lanczos3-VSST"
csv_path = "logs_cluster/{}/version_1/metrics.csv".format(model_name)
df9 = pd.read_csv(csv_path)

model_name = "realcnn-attention-bicubic-VSST"
csv_path = "logs_cluster/{}/version_1/metrics.csv".format(model_name)
df10 = pd.read_csv(csv_path)

#plot_overall_test_accuracy(df6, number_waveforms,  [i for i in range(-14, -2, 2)], "Bilinear-SST-monitor-val")
#plot_overall_test_accuracy(df7, number_waveforms,  [i for i in range(-14, -2, 2)], "Nearest-monitor-val")
plot_overall_test_accuracy(df8, number_waveforms,  [i for i in range(-14, -2, 2)], "Lanczos2-VSST-monitor-val")
plot_overall_test_accuracy(df9, number_waveforms,  [i for i in range(-14, -2, 2)], "Lanczos3-VSST-monitor-val")
plot_overall_test_accuracy(df10, number_waveforms,  [i for i in range(-14, -2, 2)], "Bicubic-VSST-monitor-val")

plt.title("Test Accuracy")
plt.show()