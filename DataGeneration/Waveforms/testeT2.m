function s = testeT2(fs, A, fc, T_pulse, Nps, Ng)

Ts = 1/fs;               % Sampling interval
t = 0:Ts:T_pulse - Ts;   % Time vector for the full pulse
T_seg = T_pulse / Ng;    % Duration of each segment
pw = T_pulse;
phi = zeros(size(t));    % Initialize phase vector

for jj = 0:Ng-1
    % Find indices corresponding to segment jj
    idx = (t >= (jj)*T_seg) & (t < (jj + 1)*T_seg);
    t_seg = t(idx) - (jj)*T_seg;
    t_seg = t_seg - t_seg(1);
    phi_quant = mod(2*pi/Nps*floor((Ng*t_seg-jj*pw)*(2*jj-Ng+1)/pw*Nps/2),2*pi);
    phi(idx) = phi_quant;
end

% Generate the complex radar waveform using a carrier and the computed phase:
s = A * exp(1i*(2*pi*fc*t + phi));
end