%%
clc
close all
clear
baseDir = 'ecg-id-database-1.0.0/Person_';
% get number of recordings for each person
numPersons = 90;
numRecs = zeros(1,numPersons);
for i = 1:90
    fullPath = strcat(baseDir,sprintf( '%02d', i),'/');
    recording = dir(fullfile(fullPath,'*.dat'));
    numRecs(i) = numel(recording);
end
[sortedRecs,sortedIndxs] = sort(numRecs,'descend');

numPersons = 10; %number of people's recodings to use
for i = 1:numPersons
    % get people with most number of recordings
    Person(i).number = sortedIndxs(i);
end
for i = 1:length(Person)
    Person(i).path = strcat(baseDir,sprintf( '%02d', Person(i).number),'/');
end

numTraining = 0;
numTesting = 0;
for i = 1:length(Person)
    recording = dir(fullfile(Person(i).path,'*.dat'));
    Person(i).numRecordings = numel(recording);
    Person(i).TrainRecordings = ceil(numel(recording)*0.7);
    Person(i).TestRecordings = numel(recording)- Person(i).TrainRecordings;
    Person(i).recordNames = {recording(:).name};
    numTraining = numTraining + Person(i).TrainRecordings;
    numTesting = numTesting + Person(i).TestRecordings;
end

plotting = 1; % turn plotting on(1) or off(0)
SamplesPerBeat = 500;
TrainingFeatures = zeros(500, 1);
TrainingLabels = zeros(1, 1);
TestingFeatures = zeros(500, 1);
TestingLabels = zeros(1, 1);
for i = 1:length(Person)
    for j = 1:length(Person(i).recordNames)
        recordPath = fullfile(Person(i).path,Person(i).recordNames{j});
        [sig, Fs, tm] = rdsamp(recordPath, 2);
        if(plotting)
            figure(1); 
            plot(tm,sig);
            title('ECG signal Before Filtering');
            xlabel('Time (sec)');
            ylabel('Voltage (mV)');
            xlim([0 4])
        end
        sig = ApplyEcgFilters(sig);
        [qrs_amp_raw,qrs_i_raw,delay] = pan_tompkin(sig,Fs,0);
        [Pidxs] = DetectPpeaks(sig,qrs_i_raw);
        %cut down data so we only have fully annotated beats (all peaks labeled)
        Pidxs = Pidxs(1:end-1);
        if(plotting)
            figure(2);
            plot(tm,sig);
            hold on;
            plot(tm(qrs_i_raw),sig(qrs_i_raw),'o','MarkerSize',10);
            plot(tm(Pidxs),sig(Pidxs),'o','MarkerSize',10);
            legend('ecg signal','R-peaks','P-peaks');
            title('Filtered ECG signal with labeled peak');
            xlabel('Time (sec)');
            ylabel('Voltage (mV)');
            hold off;
            xlim([0 4])
        end
        for k = 2:length(Pidxs)
            heartbeat = sig(Pidxs(k-1):Pidxs(k));
            resampledBeat = interp1(linspace(0,1,length(heartbeat)), heartbeat, (linspace(0,1,SamplesPerBeat)));
            low = min(resampledBeat);
            high = max(resampledBeat);
            resampledBeat = (resampledBeat-low)/(high-low);
            if(plotting)
                figure(3)
                subplot(2,1,1)
                plot(heartbeat)
                ylabel('Voltage (mV)');
                xlabel('Sample Number (n)');
                str1 =  'Heartbeat before Resampling and Normalization';
                str2 = sprintf('Subject %d, Recording %d, Beat %d\n',Person(i).number,j,k);
                title({str1;str2})
                subplot(2,1,2)
                plot(resampledBeat)
                ylabel('Voltage (mV)');
                title('Resampled and Normalized Heartbeat')
                xlabel('Sample Number (n)');
            end
            if j <= Person(i).TrainRecordings
                TrainingFeatures = [TrainingFeatures, resampledBeat'];
                TrainingLabels = [TrainingLabels,i];
            else
                TestingFeatures = [TestingFeatures, resampledBeat'];
                TestingLabels = [TestingLabels,i];
            end
        end
    end
end
TestingFeatures = TestingFeatures(:,2:end);
TestingLabels = TestingLabels(:,2:end);
TrainingFeatures = TrainingFeatures(:,2:end);
TrainingLabels = TrainingLabels( :,2:end);
save('ECGdata.mat','TestingFeatures','TestingLabels', ...
    'TrainingFeatures','TrainingLabels','Person');
