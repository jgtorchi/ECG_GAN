function fid = get_fid(a,b)
    mean1 = mean(a,2);
    mean2 = mean(b,2);
    ssdiff = sum((mean1-mean2).^2);
    cov1 = cov(a');
    cov2 = cov(b');
    smean=real(sqrtm(cov1*cov2));
    fid = ssdiff+trace(cov1+cov2-2*smean);
end