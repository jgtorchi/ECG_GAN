%% create 1D CNN
clear
clc
load('SynthData.mat')
load('ECGdata.mat')
inputSize = [size(TrainingFeatures,1) 1 1];
numClasses = max(TrainingLabels);
CNNLayers = [
    imageInputLayer(inputSize,'Normalization','none','Name','in')
    convolution2dLayer([5 1],18,'Stride',1,'Padding',[0 0],'Name','conv1')
    maxPooling2dLayer([2 1],'Name','pool1')
    reluLayer('Name','relu1')
    convolution2dLayer([5 1],18,'Stride',1,'Padding',[0 0],'Name','conv2')
    maxPooling2dLayer([2 1],'Name','pool2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(numClasses,'Name','fc1')
    softmaxLayer('Name','softmax1')
    classificationLayer('Name','class1')
    ];
lgraphCNN = layerGraph(CNNLayers);
analyzeNetwork(lgraphCNN);
options = trainingOptions('sgdm', ...
    'MaxEpochs',200, ...
    'Shuffle','every-epoch', ...
    'ValidationFrequency',30, ...
    'Verbose',true, ...
    'Plots','None');
       %training-progress
%%
TrainingCatLabels = categorical((TrainingLabels));
ReshapedTrainingFeatures = reshape(TrainingFeatures, ...
    [size(TrainingFeatures,1),1,1,size(TrainingFeatures,2)]);
ReshapedTestingFeatures = reshape(TestingFeatures, ...
        [size(TestingFeatures,1),1,1,size(TestingFeatures,2)]);
numRuns = 10;
predClasses = [];
for i=1:numRuns
    Net = trainNetwork(ReshapedTrainingFeatures,TrainingCatLabels,CNNLayers,options);
    YPred = predict(Net,ReshapedTestingFeatures);
    [~, tempPredClasses] = max(YPred,[],2);
    predClasses = [predClasses;tempPredClasses];
end
%%
TestingLabels = repmat(TestingLabels,1,numRuns);
numCorrect = sum(predClasses'==TestingLabels);
OverallAccuracy = numCorrect/length(TestingLabels);
figure(1)
confusionchart(TestingLabels,predClasses)
numClasses = max(predClasses);
% calculate stats for augmented dataset
TP = zeros(1,numClasses);
FP = zeros(1,numClasses);
TN = zeros(1,numClasses);
FN = zeros(1,numClasses);
Sensitivity = zeros(1,numClasses);
Specificity = zeros(1,numClasses);
Precision = zeros(1,numClasses);
Accuracy = zeros(1,numClasses);
for i = 1:numClasses
    TP(i) = sum(and((predClasses' == i),(TestingLabels==i)));
    FP(i) = sum(and((predClasses' == i),~(TestingLabels==i)));
    TN(i) = sum(and(~(predClasses' == i),~(TestingLabels==i)));
    FN(i) = sum(and(~(predClasses' == i),(TestingLabels==i)));
    Sensitivity(i) = TP(i)*100/(TP(i)+FN(i));
    Specificity(i) = TN(i)*100/(TN(i)+FP(i));
    Precision(i) = TP(i)*100/(TP(i)+FP(i));
    Accuracy(i) = 100*(TP(i)+TN(i))/(TP(i)+TN(i)+FP(i)+FN(i));
end
classes = 1:numClasses;
subjects = [Person(classes).number];
varNames = {'Subject','sensitivity','specificity','precision','accuracy'};
figure(2)
T = table(subjects',Sensitivity',Specificity',Precision',Accuracy','VariableNames',varNames);
uitable('Data',T{:,:},'ColumnName',T.Properties.VariableNames,...
    'RowName',T.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);
%% Create and train net with augmented Dataset
TrainingCatLabels = categorical(([TrainingLabels,double(synthLabels)]));
TrainingFeatures = [TrainingFeatures,smoothedSignal];
ReshapedTrainingFeatures = reshape(TrainingFeatures, ...
    [size(TrainingFeatures,1),1,1,size(TrainingFeatures,2)]);
ReshapedTestingFeatures = reshape(TestingFeatures, ...
    [size(TestingFeatures,1),1,1,size(TestingFeatures,2)]);
predClasses = [];
for i=1:numRuns
    augmentedNet = trainNetwork(ReshapedTrainingFeatures,TrainingCatLabels,CNNLayers,options);
    YPred = predict(augmentedNet,ReshapedTestingFeatures);
    [~, tempPredClasses] = max(YPred,[],2);
    predClasses = [predClasses;tempPredClasses];
end
numCorrect = sum(predClasses'==TestingLabels);
aOverallAccuracy = numCorrect/length(TestingLabels);
figure(3)
confusionchart(TestingLabels,predClasses)
numClasses = max(predClasses);
% calculate stats for augmented dataset
aTP = zeros(1,numClasses);
aFP = zeros(1,numClasses);
aTN = zeros(1,numClasses);
aFN = zeros(1,numClasses);
aSensitivity = zeros(1,numClasses);
aSpecificity = zeros(1,numClasses);
aPrecision = zeros(1,numClasses);
aAccuracy = zeros(1,numClasses);
for i = 1:numClasses
    aTP(i) = sum(and((predClasses' == i),(TestingLabels==i)));
    aFP(i) = sum(and((predClasses' == i),~(TestingLabels==i)));
    aTN(i) = sum(and(~(predClasses' == i),~(TestingLabels==i)));
    aFN(i) = sum(and(~(predClasses' == i),(TestingLabels==i)));
    aSensitivity(i) = aTP(i)*100/(aTP(i)+aFN(i));
    aSpecificity(i) = aTN(i)*100/(aTN(i)+aFP(i));
    aPrecision(i) = aTP(i)*100/(aTP(i)+aFP(i));
    aAccuracy(i) = 100*(aTP(i)+aTN(i))/(aTP(i)+aTN(i)+aFP(i)+aFN(i));
end
classes = 1:numClasses;
subjects = [Person(classes).number];
varNames = {'Subject','sensitivity','specificity','precision','accuracy'};
figure(4)
T = table(subjects',aSensitivity',aSpecificity',aPrecision',aAccuracy','VariableNames',varNames);
uitable('Data',T{:,:},'ColumnName',T.Properties.VariableNames,...
    'RowName',T.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);
%%
dOverallAccuracy = aOverallAccuracy-OverallAccuracy;
dAccuracy = aAccuracy-Accuracy;
dSensitivity = aSensitivity-Sensitivity;
dSpecificity = aSpecificity-Specificity;
dPrecision = aPrecision-Precision;
classes = 1:numClasses;
subjects = [Person(classes).number];
figure(5)
T = table(subjects',dSensitivity',dSpecificity',dPrecision',dAccuracy','VariableNames',varNames);
uitable('Data',T{:,:},'ColumnName',T.Properties.VariableNames,...
    'RowName',T.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);
%% get number of 
classes = 1:numClasses;
numTestSamples = zeros(1,numClasses);
for i=1:numClasses
    numTestSamples(i)=sum(TestingLabels==i);
end
varNames = {'Classes','numTrainSamples','numSynthSamples',...
    'TotalTrainSamples','numTestSamples'};
figure(6)
T = table(classes',numSamplesPerClass',numSynthPerClass',...
    (numSamplesPerClass+numSynthPerClass)',numTestSamples',...
    'VariableNames',varNames);
uitable('Data',T{:,:},'ColumnName',T.Properties.VariableNames,...
    'RowName',T.Properties.RowNames,'Units', 'Normalized', 'Position',[0, 0, 1, 1]);
%%
avgSensitivity = mean(Sensitivity);
avgSpecificity = mean(Specificity);
avgPrecision = mean(Precision);

aAvgSensitivity = mean(aSensitivity);
aAvgSpecificity = mean(aSpecificity);
aAvgPrecision = mean(aPrecision);
