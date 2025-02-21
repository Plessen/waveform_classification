function [output,F,T] = CWD(x0,fs,sigma)

if ( all(imag(x0)==0)); x0 = hilbert(x0); end
lx = length(x0);
nfft = 1024;
nfftby2 = nfft/2;

x = zeros(nfft,1);
x(1:lx) = x0(:); cx = conj(x);
wx = zeros(nfft,nfft);
L1 = lx - 1;

% --------- compute r(tau,t) = cx(t-tau/2) * x(t+tau/2) -------------
for n=0:L1
    indm = max(-n,-L1+n) : min(n,L1-n);
    indy = indm + (indm < 0) * nfft ;   % output indices y(m;n)
    y = zeros(nfft,1);
    y(indy + 1) = x(n+indm + 1) .* cx(n-indm + 1);
    wx(:,n+1)   = y;
end

wx  = ifft(wx.').';

win = [(1:nfft)-nfft/2-1]' * [(1:nfft)-nfft/2-1] / nfft;
win = fftshift( exp (- win.^2 / sigma) );
wx  = wx .* win;

% -------------- WD(t,f) = FT A(theta,tau) -------------------
wx = fft2(wx);         % fft along both time and frequency
wx  = real(wx(:,1:lx));

% ----------- WD(f,t) = FT (tau-->f) r(tau,t) ---------------------
%output = real(wx([nfftby2+1:nfft,1:nfftby2],:));
output = real(wx);
F = [-nfftby2:nfftby2-1] / (2*nfft) * fs;
T = (1:lx)/fs;