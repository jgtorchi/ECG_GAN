n = 300;
hpf = fir1(n,0.002,'high',kaiser(n+1,0.5));
bsf = fir1(n,[0.2380 0.2420],'stop',kaiser(n+1,0.5));
lpf = fir1(n,0.4,'low',kaiser(n+1,0.5));
[h0,w] = freqz(hpf,1,n);
[h1,w] = freqz(bsf,1,n);
[h2,w] = freqz(lpf,1,n);
fs = 500;
w = linspace(0,fs*0.5,length(h0));
figure(1)
subplot(4,1,1)
plot(w,abs(h0))
title('High Pass Filter')
subplot(4,1,2)
plot(w,abs(h1))
title('Band Stop Filter')
subplot(4,1,3)
plot(w,abs(h2))
title('Low Pass Filter')
subplot(4,1,4)
plot(w,abs(h0.*h1.*h2))
xlabel('Frequency (Hz)')
ylabel('Magnitude Response')
title('Cascaded Filter Response')