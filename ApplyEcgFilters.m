function FilteredEcg = ApplyEcgFilters(Ecg)
    n = 300;
    hpf = fir1(n,0.002,'high',kaiser(n+1,0.5));
    FilteredEcg = filter(hpf,1,Ecg);
    bsf = fir1(n,[0.2380 0.2420],'stop',kaiser(n+1,0.5));
    FilteredEcg = filter(bsf,1,FilteredEcg);
    lpf = fir1(n,0.4,'low',kaiser(n+1,0.5));
    FilteredEcg = filter(lpf,1,FilteredEcg);
    windowWidth = 9; 
    kernel = ones(windowWidth,1) / windowWidth;
    FilteredEcg = filter(kernel, 1, FilteredEcg);
end