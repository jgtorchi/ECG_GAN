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
for i = 1:maxLabel
    if(numSamplesPerClass(i)<(0.5*maxSample))
        numGanTrainingPerClass(i) = sum(numSamplesPerClass(i));
    end
end
GanTrainingFeatures = zeros(size(TrainingFeatures,1),sum(numGanTrainingPerClass));
GanTrainingLabels = zeros(1,sum(numGanTrainingPerClass));
high = 0;
for i=1:maxLabel
    if (numGanTrainingPerClass(i)>0)
        low = high+1;
        high = low + numGanTrainingPerClass(i) - 1;
        GanTrainingFeatures(:,low:high) = TrainingFeatures(:,TrainingLabels==i);
        GanTrainingLabels(:,low:high) = i;
    end
end
GanTrainingLabels = GanTrainingLabels-3;
%% create Generator Network

numFilters = 64;
numLatentInputs = 100;
projectionSize = [1 1 size(GanTrainingFeatures,1)];
%projectionSize = [1 1 numLatentInputs];
numClasses = length(unique(GanTrainingLabels));
embeddingDimension = 100;

layersGenerator = [
    imageInputLayer([1 1 numLatentInputs],'Normalization','none','Name','in')
    projectAndReshapeLayer(projectionSize,numLatentInputs,'proj');
    concatenationLayer(1,2,'Name','cat');
    transposedConv2dLayer([7 1],32*numFilters,'Stride',1,'Name','tconv1')
    batchNormalizationLayer('Name','bn1','Epsilon',5e-5)
    reluLayer('Name','relu1')
    transposedConv2dLayer([5 1],16*numFilters,'Stride',1,'Cropping',[0 0],'Name','tconv2')
    batchNormalizationLayer('Name','bn2','Epsilon',5e-5)
    reluLayer('Name','relu2')
    transposedConv2dLayer([5 1],8*numFilters,'Stride',2,'Cropping',[0 0],'Name','tconv3')
    batchNormalizationLayer('Name','bn3','Epsilon',5e-5)
    reluLayer('Name','relu3')
    transposedConv2dLayer([5 1],4*numFilters,'Stride',2,'Cropping',[0 0],'Name','tconv4')
    batchNormalizationLayer('Name','bn4','Epsilon',5e-5)
    reluLayer('Name','relu4')
    transposedConv2dLayer([5 1],2*numFilters,'Stride',2,'Cropping',[0 0],'Name','tconv5')
    batchNormalizationLayer('Name','bn5','Epsilon',5e-5)
    reluLayer('Name','relu5')
    transposedConv2dLayer([5 1],1*numFilters,'Stride',2,'Cropping',[0 0],'Name','tconv6')
    batchNormalizationLayer('Name','bn6','Epsilon',5e-5)
    reluLayer('Name','relu6')
    transposedConv2dLayer([5 1],1,'Stride',2,'Cropping',[0 0],'Name','tconv7')
    batchNormalizationLayer('Name','bn7','Epsilon',5e-5)
    reluLayer('Name','relu7')
    projectAndReshapeLayer([500 1 1], 477, 'proj1')
    sigmoidLayer('Name','sig1')
    ];


lgraphGenerator = layerGraph(layersGenerator);

layers = [
    imageInputLayer([1 1],'Name','labels','Normalization','none')
    embedAndReshapeLayer(projectionSize,embeddingDimension,numClasses,'emb')];

lgraphGenerator = addLayers(lgraphGenerator,layers);
lgraphGenerator = connectLayers(lgraphGenerator,'emb','cat/in2');
dlnetGenerator = dlnetwork(lgraphGenerator);
analyzeNetwork(lgraphGenerator);
%% create Discriminator Network

scale = 0.2;
inputSize = [size(GanTrainingFeatures,1) 1 1];

layersDiscriminator = [
    imageInputLayer(inputSize,'Normalization','none','Name','in')
    concatenationLayer(3,2,'Name','cat')
    convolution2dLayer([7 1],numFilters,'Stride',2,'Padding',[0 0],'Name','conv1')
    leakyReluLayer(scale,'Name','lrelu1')
    convolution2dLayer([5 1],2*numFilters,'Stride',2,'Padding',[0 0],'Name','conv2')
    leakyReluLayer(scale,'Name','lrelu2')
    convolution2dLayer([5 1],4*numFilters,'Stride',2,'Padding',[0 0],'Name','conv3')
    leakyReluLayer(scale,'Name','lrelu3')
    convolution2dLayer([5 1],8*numFilters,'Stride',2,'Padding',[0 0],'Name','conv4')
    leakyReluLayer(scale,'Name','lrelu4')
    convolution2dLayer([5 1],16*numFilters,'Name','conv5')
    leakyReluLayer(scale,'Name','lrelu5')
    fullyConnectedLayer(1,'Name','fc1')
    %sigmoidLayer('Name','sig1')
    ];

lgraphDiscriminator = layerGraph(layersDiscriminator);

layers = [
    imageInputLayer([1 1],'Name','labels','Normalization','none')
    embedAndReshapeLayer(inputSize,embeddingDimension,numClasses,'emb')];

lgraphDiscriminator = addLayers(lgraphDiscriminator,layers);
lgraphDiscriminator = connectLayers(lgraphDiscriminator,'emb','cat/in2');
analyzeNetwork(lgraphDiscriminator);
dlnetDiscriminator = dlnetwork(lgraphDiscriminator);
%% specify training parameters
params.numLatentInputs = numLatentInputs;
params.numClasses = numClasses;
params.sizeData = [inputSize size(GanTrainingFeatures,2)];
params.numEpochs = 500;
params.miniBatchSize = 64;

% Specify the options for Adam optimizer
params.learnRate = 0.0002;
params.gradientDecayFactor = 0.5;
params.squaredGradientDecayFactor = 0.999;
%%
executionEnvironment = "gpu";
params.executionEnvironment = executionEnvironment;
%% Train the CGAN
[dlnetGenerator,dlnetDiscriminator] = trainGAN(dlnetGenerator, ...
    dlnetDiscriminator,GanTrainingFeatures,GanTrainingLabels,params);
%% save network
save('GAN14.mat','dlnetGenerator','dlnetDiscriminator', ...
    'lgraphGenerator','lgraphDiscriminator','params', ...
    'GanTrainingFeatures','GanTrainingLabels')