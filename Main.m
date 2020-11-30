%%
clc
close all
clear
Person(1).number = 1;
Person(2).number = 2;
Person(3).number = 9;
Person(4).number = 52;
Person(5).number = 72;
baseDir = 'ecg-id-database-1.0.0/Person_';
for i = 1:length(Person)
    Person(i).path = strcat(baseDir,sprintf( '%02d', Person(i).number),'/');
end

numTraining = 0;
numTesting = 0;
for i = 1:length(Person)
    recording = dir(fullfile(Person(i).path,'*.dat'));
    Person(i).numRecordings = numel(recording);
    Person(i).TrainRecordings = ceil(numel(recording)*0.6);
    Person(i).TestRecordings = numel(recording)- Person(i).TrainRecordings;
    Person(i).recordNames = {recording(:).name};
    numTraining = numTraining + Person(i).TrainRecordings;
    numTesting = numTesting + Person(i).TestRecordings;
end

SamplesPerBeat = 500;
TrainingFeatures = zeros(500, 1);
TrainingLabels = zeros(1, 1);
TestingFeatures = zeros(500, 1);
TestingLabels = zeros(1, 1);
for i = 1:length(Person)
    for j = 1:length(Person(i).recordNames)
        recordPath = fullfile(Person(i).path,Person(i).recordNames{j});
        [sig, Fs, tm] = rdsamp(recordPath, 2);
        figure(1);
        plot(tm,sig);
        title('ECG signal Before Filtering');
        xlabel('Time (sec)');
        ylabel('Voltage (mV)');
        sig = ApplyEcgFilters(sig);
        [qrs_amp_raw,qrs_i_raw,delay] = pan_tompkin(sig,Fs,0);
        [Pidxs] = DetectPpeaks(sig,qrs_i_raw);
        %cut down data so we only have fully annotated beats (all peaks labeled)
        Pidxs = Pidxs(1:end-1);
        figure(2);
        plot(tm,sig);
        hold on;
        plot(tm(Pidxs),sig(Pidxs),'o','MarkerSize',10);
        legend('ecg signal','P-peaks');
        title('Filtered ECG signal with labeled peak');
        xlabel('Time (sec)');
        ylabel('Voltage (mV)');
        hold off;
        for k = 2:length(Pidxs)
            heartbeat = sig(Pidxs(k-1):Pidxs(k));
            resampledBeat = interp1(linspace(0,1,length(heartbeat)), heartbeat, (linspace(0,1,SamplesPerBeat)));
            figure(3)
            subplot(2,1,1)
            plot(heartbeat)
            ylabel('Voltage (mV)');
            xlabel('Sample Number (n)');
            title('Heartbeat before Resampling')
            subplot(2,1,2)
            plot(resampledBeat)
            ylabel('Voltage (mV)');
            title('Resampled Heartbeat')
            xlabel('Sample Number (n)');
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
