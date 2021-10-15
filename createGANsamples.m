clc
clear
close all
rng default
load('GAN6.mat')
load('ECGdata.mat')
maxLabel = max(TrainingLabels);
numSynthPerClass = zeros(1,maxLabel);
numSamplesPerClass = zeros(1,maxLabel);
for i = 1:maxLabel
    numSamplesPerClass(i) = sum(TrainingLabels == i);
end
maxSample = max(numSamplesPerClass);
for i = 1:maxLabel
    if(numSamplesPerClass(i)<(0.5*maxSample))
        numSynthPerClass(i) = floor(sum(numSamplesPerClass(i))*0.7);
    end
end
numTests = sum(numSynthPerClass);
TNew = zeros(1,1,1,numTests,'single');
ZNew = randn(1,1,params.numLatentInputs,numTests,'single');
% assign Tnew (labels of synthetic data)
high = 0;
for i = 1:maxLabel
    low = high+1;
    high = low + numSynthPerClass(i) - 1;
    TNew(1,1,1,((low):(high))) = single(i);
end
TNew = TNew -3;
%%
dlTNew = dlarray(TNew,'SSCB');
dlZNew = dlarray(ZNew,'SSCB');
executionEnvironment = "gpu";
if executionEnvironment == "gpu"
    dlZNew = gpuArray(dlZNew);
    dlTNew = gpuArray(dlTNew);
end
dlXGeneratedNew = predict(dlnetGenerator,dlZNew,dlTNew);
dlXGeneratedNew = sigmoid(dlXGeneratedNew);
idxGenerated = 1:numTests;
idxReal = numTests+1:numTests+size(flow,2);
XGeneratedNew = squeeze(extractdata(gather(dlXGeneratedNew)));
%% smooth signal with Savitzky-Golay filter
order = 25;
filtLength = 51;
smoothedSignal = sgolayfilt(double(XGeneratedNew),order,filtLength);
%% normalize generated heartbeats from zero to one
synthLabels = squeeze(TNew(1,1,1,:));
synthLabels = synthLabels+3;
offset = 15;
for i = 1:size(smoothedSignal,2)
    ecg = smoothedSignal(:,i);
    low = min(ecg(1:end-offset));
    high = max(ecg(1:end-offset));
    ecg = (ecg-low)/(high-low);
    plot(ecg)
    ecgs = TrainingFeatures(:,TrainingLabels==synthLabels(i));
    r = randi([1 size(ecgs,2)],1,1);
    ecgs = ecgs(:,r);
    ecg(end-offset:end) = ecgs(end-offset:end)+ ecg(end-offset-1)...
        - ecgs(end-offset-1);
    hold on
    plot(ecg)
    hold off
    xlabel('Sample Number (n)')
    ylabel('Normalized Amplitude')
    legend('GAN Generated ECG','Modified GAN ECG')
    title('Modifying end of GAN signals')
    smoothedSignal(:,i) = ecg;
end
%% Evaluate Generated Signals using various metrics
labels = unique(synthLabels);
fid = zeros(1,length(labels));
ed = zeros(1,length(labels));
DTW = zeros(1,length(labels));
for i = 1:length(labels)
    synthEcgs = smoothedSignal(:,synthLabels==labels(i));
    ecgs = TrainingFeatures(:,TrainingLabels==labels(i));
    % get frechet inception distance
    fid(i) = get_fid(synthEcgs, ecgs);
    ed(i) = avg_ed(ecgs,synthEcgs);
    DTW(i) = avg_dtw(ecgs,synthEcgs);
end
%% plot ECGs
for i = 1:length(labels)
    figure(i)
    ecgs = TrainingFeatures(:,TrainingLabels==labels(i));
    r = randi([1 size(ecgs,2)],1,1);
    plot(ecgs(:,r))
    hold on
    ecgs = smoothedSignal(:,synthLabels==labels(i));
    r = randi([1 size(ecgs,2)],1,1);
    plot(ecgs(:,r))
    hold off
    legend('Original','Synthetic')
    ylabel('Normalized Amplitude')
    xlabel('Sample Number (n)')
    title('Original and Unfiltered Synthetic ECG')
end
disp(mean(fid))
%%
synthLabels = synthLabels';
save('SynthData.mat','smoothedSignal','synthLabels',...
    'numSamplesPerClass','numSynthPerClass');
