fs = 1000;
A = 1;
phaseCode = [0 0 0 0 0 1 1 0 0 1 0 1 0]*pi;
barker_signal = type_Barker(20, fs, A, fs / 5, phaseCode);
T = (length(barker_signal) / fs);

[afmag,delay] = ambgfun(barker_signal, fs, 1 / T,"Cut","Doppler");
plot(delay / (T / 13), afmag);
xlabel('\tau / t_b', 'FontSize', 40);
ylabel("Autocorrelation", 'FontSize', 40);
set(gca, 'FontSize', 30);
xlim([-13,13]);
