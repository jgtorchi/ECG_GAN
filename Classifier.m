%% create 1D CNN
clear
clc
load('SynthData.mat')
load('ECGdata.mat')
inputSize = [size(TrainingFeatures,1) 1 1];
numClasses = max(TrainingLabels);
CNNLayers = [
    imageInputLayer(inputSize,'Normalization','none','Name','in')
    convolution2dLayer([7 1],18,'Stride',1,'Padding',[0 0],'Name','conv1')
    maxPooling2dLayer([2 1],'Name','pool1')
    reluLayer('Name','relu1')
    convolution2dLayer([7 1],18,'Stride',1,'Padding',[0 0],'Name','conv2')
    maxPooling2dLayer([2 1],'Name','pool2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(numClasses,'Name','fc1')
    softmaxLayer('Name','softmax1')
    classificationLayer('Name','class1')
    ];
%lgraphCNN = layerGraph(CNNLayers);
%analyzeNetwork(lgraphCNN);
%options = trainingOptions('adam');
options = trainingOptions('sgdm', ...
    'MaxEpochs',200, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','training-progress');
augmented = 1;
if(augmented)
    TrainingCatLabels = categorical(([TrainingLabels,double(synthLabels)]));
    TrainingFeatures = [TrainingFeatures,smoothedSignal];
    ReshapedTrainingFeatures = reshape(TrainingFeatures, ...
        [size(TrainingFeatures,1),1,1,size(TrainingFeatures,2)]);
else
    TrainingCatLabels = categorical((TrainingLabels));
    ReshapedTrainingFeatures = reshape(TrainingFeatures, ...
        [size(TrainingFeatures,1),1,1,size(TrainingFeatures,2)]);
end
convnet = trainNetwork(ReshapedTrainingFeatures,TrainingCatLabels,CNNLayers,options);
%%
ReshapedTestingFeatures = reshape(TestingFeatures, ...
    [size(TestingFeatures,1),1,1,size(TestingFeatures,2)]);
YPred = predict(convnet,ReshapedTestingFeatures);
[~, predClasses] = max(YPred,[],2);
numCorrect = sum(predClasses'==TestingLabels);
percentCorrect = numCorrect/length(TestingLabels);