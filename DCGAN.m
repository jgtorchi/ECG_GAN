clc
close all
clear
%% load data
load('ECGdata.mat')
%% setup GAN training data
maxLabel = max(TrainingLabels);
numSamplesPerClass = zeros(1,maxLabel);
for i = 1:maxLabel
    numSamplesPerClass(i) = sum(TrainingLabels == i);
end
maxSample = max(numSamplesPerClass);
numGanTrainingPerClass = zeros(1,maxLabel);
numSynthPerClass = zeros(1,maxLabel);
for i = 1:maxLabel
    if(numSamplesPerClass(i)<(0.5*maxSample))
        numGanTrainingPerClass(i) = sum(numSamplesPerClass(i));
        numSynthPerClass = floor(numGanTrainingPerClass*0.5);
    end
end
%% create Generator Network

numFilters = 64;
numLatentInputs = 100;
projectionSize = [4 1 size(TrainingFeatures,1)];
embeddingDimension = 100;

layersGenerator = [
    imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','in')
    projectAndReshapeLayer(projectionSize,numLatentInputs,'proj');
    transposedConv2dLayer([5 1],8*numFilters,'Name','tconv1')
    batchNormalizationLayer('Name','bn1','Epsilon',5e-5)
    reluLayer('Name','relu1')
    transposedConv2dLayer([10 1],4*numFilters,'Stride',3,'Cropping',[1 0],'Name','tconv2')
    batchNormalizationLayer('Name','bn2','Epsilon',5e-5)
    reluLayer('Name','relu2')
    transposedConv2dLayer([12 1],2*numFilters,'Stride',4,'Cropping',[1 0],'Name','tconv3')
    batchNormalizationLayer('Name','bn3','Epsilon',5e-5)
    reluLayer('Name','relu3')
    transposedConv2dLayer([7 1],numFilters,'Stride',2,'Cropping',[1 0],'Name','tconv4')
    batchNormalizationLayer('Name','bn4','Epsilon',5e-5)
    reluLayer('Name','relu4')
    transposedConv2dLayer([8 1],1,'Stride',2,'Cropping',[0 0],'Name','tconv5')
    %batchNormalizationLayer('Name','bn5','Epsilon',5e-5)
    %reluLayer('Name','relu5')
    %projectAndReshapeLayer([500 1 1], 500, 'proj1')
    ];

lgraphGenerator = layerGraph(layersGenerator);

dlnetGenerator = dlnetwork(lgraphGenerator);
analyzeNetwork(lgraphGenerator);
%% create Discriminator Network

scale = 0.2;
inputSize = [size(TrainingFeatures,1) 1 1];

layersDiscriminator = [
    imageInputLayer(inputSize,'Normalization','none','Name','in')
    convolution2dLayer([17 1],8*numFilters,'Stride',2,'Padding',[1 0],'Name','conv1')
    leakyReluLayer(scale,'Name','lrelu1')
    convolution2dLayer([16 1],4*numFilters,'Stride',4,'Padding',[1 0],'Name','conv2')
    leakyReluLayer(scale,'Name','lrelu2')
    convolution2dLayer([16 1],2*numFilters,'Stride',2,'Padding',[1 0],'Name','conv3')
    leakyReluLayer(scale,'Name','lrelu3')
    convolution2dLayer([8 1],numFilters,'Stride',2,'Padding',[0 0],'Name','conv4')
    leakyReluLayer(scale,'Name','lrelu4')
    convolution2dLayer([8 1],1,'Name','conv5')];

lgraphDiscriminator = layerGraph(layersDiscriminator);


%analyzeNetwork(lgraphDiscriminator);
dlnetDiscriminator = dlnetwork(lgraphDiscriminator);
%% specify training parameters
params.numLatentInputs = numLatentInputs;
params.numClasses = 1;
params.numEpochs = 100;

% Specify the options for Adam optimizer
params.learnRate = 0.0001;
params.gradientDecayFactor = 0.5;
params.squaredGradientDecayFactor = 0.999;
%%
executionEnvironment = "gpu";
params.executionEnvironment = executionEnvironment;
%% Train the CGAN
labels = unique(TrainingLabels);
first = 1;
for i =1:length(labels)
    if (numGanTrainingPerClass(i)>0)
        GanTrainingFeatures = TrainingFeatures(:, TrainingLabels==1);
        GanTrainingLabels = TrainingLabels(:, TrainingLabels==1);
        params.sizeData = [inputSize size(GanTrainingFeatures,2)];
        params.miniBatchSize = 6;
        [dlnetGenerator,dlnetDiscriminator] = trainUncondGAN(dlnetGenerator, ...
            dlnetDiscriminator,GanTrainingFeatures,GanTrainingLabels,params);
        % generate synthetic ECGs
        ZNew = randn(1,1,params.numLatentInputs,numSynthPerClass(i),'single');
        dlZNew = dlarray(ZNew,'SSCB');
        if executionEnvironment == "gpu"
            dlZNew = gpuArray(dlZNew);
        end
        dlXGeneratedNew = predict(dlnetGenerator,dlZNew);
        XGeneratedNew = squeeze(extractdata(gather(dlXGeneratedNew)));
        figure(1)
        plot(GanTrainingFeatures(:,1))
        hold on
        plot(XGeneratedNew(:,1))
        hold off
        legend('Original','Synthetic')
        ylabel('Normalized Amplitude')
        xlabel('Sample Number (n)')
        title('Original and Unfiltered Synthetic ECG')
        if(first)
            synthECGs = XGeneratedNew;
            synthLabels = ones(1,size(XGeneratedNew,2))*i;
            first = 0;
        else
            synthECGs = [synthECGs,XGeneratedNew];
            synthLabels = [synthLabels, (ones(1,size(XGeneratedNew,2))*i)];
        end
        close all
    end
end
%% smooth synthetic ECGs with Savitzky-Golay filter
order = 15;
filtLength = 31;
smoothedSignal = sgolayfilt(double(synthECGs),order,filtLength);
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
    legend('Original Syth ECG','Modified Synth ECG')
    smoothedSignal(:,i) = ecg;
end
%% Evaluate Generated Signals using various metrics
labels = unique(synthLabels);
fid = zeros(1,length(labels));
for i = 1:length(labels)
    sythEcgs = smoothedSignal(:,synthLabels==labels(i));
    ecgs = TrainingFeatures(:,TrainingLabels==labels(i));
    % get frechet inception distance
    fid(i) = get_fid(sythEcgs, ecgs);
    ed(i) = avg_ed(ecgs,synthEcgs);
    DTW(i) = avg_dtw(ecgs,synthEcgs);
end
%% display all ECGs
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
%% save network and sythetic ECGs
save('GAN8.mat','dlnetGenerator','dlnetDiscriminator', ...
    'lgraphGenerator','lgraphDiscriminator','params', ...
    'GanTrainingFeatures','GanTrainingLabels')
save('SynthData2.mat','smoothedSignal','synthLabels',...
    'numSamplesPerClass','numSynthPerClass');