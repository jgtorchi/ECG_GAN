clc
clear
rng default
load('GAN2.mat')
load('ECGdata.mat')
maxLabel = max(TrainingLabels);
numTestsPerClass = zeros(1,maxLabel);
for i = 1:maxLabel
    numTestsPerClass(i) = floor(sum(TrainingLabels == i)*0.3);
end
numTests = sum(numTestsPerClass);
TNew = zeros(1,1,1,numTests,'single');
ZNew = randn(1,1,params.numLatentInputs,numTests,'single');
% assign Tnew (labels of synthetic data)
high = 0;
for i = 1:maxLabel
    low = high+1;
    high = low + numTestsPerClass(i) - 1;
    TNew(1,1,1,((low):(high))) = single(i);
end
%%
dlTNew = dlarray(TNew,'SSCB');
dlZNew = dlarray(ZNew,'SSCB');
executionEnvironment = "gpu";
if executionEnvironment == "gpu"
    dlZNew = gpuArray(dlZNew);
    dlTNew = gpuArray(dlTNew);
end
dlXGeneratedNew = predict(dlnetGenerator,dlZNew,dlTNew);
idxGenerated = 1:numTests;
idxReal = numTests+1:numTests+size(flow,2);
XGeneratedNew = squeeze(extractdata(gather(dlXGeneratedNew)));
%% smooth signal with Savitzky-Golay filter
order = 15;
length = 31;
smoothedSignal = sgolayfilt(double(XGeneratedNew),order,length);
%% Evaluate Generated Signals using various metrics
synthLabels = squeeze(TNew(1,1,1,:));
y = smoothedSignal(:,(synthLabels == 1)');
x = repmat((1:size(y,1))',1,size(y,2));
xy = [x,y];
figure(1)
scatter(xy(:,1),xy(:,2));

sig = TrainingFeatures(:,(TrainingLabels == 3)');
avgSig = sum(sig,2)/size(sig,2);
plot(avgSig)

%%
figure(1)
plot(TrainingFeatures(:,1))
hold on
plot(smoothedSignal(:,30))
%plot(XGeneratedNew(:,21))
hold off
legend('Original','Synthetic')
ylabel('Voltage (mV)')
xlabel('Sample Number (n)')
title('Original and Unfiltered Synthetic ECG')
%%
synthLabels = synthLabels';
save('SynthData.mat','smoothedSignal','synthLabels');
