clc;
clearvars;
close all;

% dataset
fldr = "C:\Users\kayja\Documents\sem5_projects\ASP\VoxCeleb_gender";
ads = audioDatastore(fldr, "IncludeSubfolders", true);
ads.Labels = categorical(extractBetween(ads.Files, fullfile(fldr,filesep), filesep));
[adsTrain, adsTest] = splitEachLabel(ads, 0.8);

[audioIn, dsInfo] = read(adsTrain);
Fs = dsInfo.SampleRate;

reset(adsTrain);

% data preprocessing
frameDuration = 200e-3;
overlapDuration = 40e-3;
frameLength = floor(Fs*frameDuration); 
overlapLength = round(Fs*overlapDuration);

pFlag = ~isempty(ver("parallel"));

adsTrainTransform = transform(adsTrain, @(x){preprocessAudioData(x, frameLength, overlapLength, Fs)});
XTrain = readall(adsTrainTransform, UseParallel = pFlag);

chunksPerFile = cellfun(@(x)size(x,4), XTrain);
TTrain = repelem(adsTrain.Labels, chunksPerFile, 1);

XTrain = cat(4, XTrain{:});
adsTestTransform = transform(adsTest, @(x){preprocessAudioData(x, frameLength, overlapLength, Fs)});
XTest = readall(adsTestTransform, UseParallel=true);
chunksPerFile = cellfun(@(x)size(x, 4), XTest);
TTest = repelem(adsTest.Labels, chunksPerFile, 1);
XTest = cat(4, XTest{:});

% standard CNN
numFilters = 80;
filterLength = 251;
numSpeakers = numel(unique(removecats(ads.Labels)));

layers = [ 
    imageInputLayer([1, frameLength, 1])
    
    % First convolutional layer
    
    convolution2dLayer([1, filterLength], numFilters)     % this layer is replaced by SincNet layer
    batchNormalizationLayer
    leakyReluLayer(0.2)
    maxPooling2dLayer([1, 3])
    
    % This layer is followed by 2 convolutional layers
    
    convolution2dLayer([1, 5], 60)
    batchNormalizationLayer
    leakyReluLayer(0.2)
    maxPooling2dLayer([1, 3])
    
    convolution2dLayer([1, 5], 60)
    batchNormalizationLayer
    leakyReluLayer(0.2)
    maxPooling2dLayer([1, 3])

    % This is followed by 3 fully-connected layers
    
    fullyConnectedLayer(256)
    batchNormalizationLayer
    leakyReluLayer(0.2)
    
    fullyConnectedLayer(256)
    batchNormalizationLayer
    leakyReluLayer(0.2)

    fullyConnectedLayer(256)
    batchNormalizationLayer
    leakyReluLayer(0.2)

    fullyConnectedLayer(numSpeakers)
    softmaxLayer
    classificationLayer];

% training the CNN
numEpochs = 1;
miniBatchSize = 128;
validationFrequency = floor(numel(TTrain)/miniBatchSize);

options = trainingOptions("adam", ...
    Shuffle = "every-epoch", ...
    MiniBatchSize = miniBatchSize, ...
    Plots = "training-progress", ...
    Verbose = true, ...
    VerboseFrequency = 212, ...
    MaxEpochs = numEpochs, ...
    Shuffle = "once", ...
    InitialLearnRate = 0.003, ...
    LearnRateSchedule = "piecewise", ...
    LearnRateDropPeriod = 2, ...
    LearnRateDropFactor = 0.25, ...
    ValidationData = {XTest, categorical(TTest)}, ...
    ValidationFrequency = validationFrequency);

[convNet, convNetInfo] = trainNetwork(XTrain, TTrain, layers, options);

% frequency response of the CNN
F = squeeze(convNet.Layers(2, 1).Weights);
H = zeros(size(F));
Freq = zeros(size(F));

for ii = 1:size(F, 2)
    [h, f] = freqz(F(:, ii), 1, 251, Fs);
    H(:, ii) = abs(h);
    Freq(:, ii) = f;
end

idx = linspace(1, size(F, 2), 9);
idx = round(idx);

figure
for jj = 1:9
   subplot(3, 3, jj)
   plot(Freq(:, idx(jj)), H(:, idx(jj)))
   sgtitle("Frequency Response of Learned Standard CNN Filters")
   xlabel("Frequency (Hz)")
end

% SincNet
numFilters = 80;
filterLength = 251;
numChannels = 1; 
name = "sinc";

sNL = sincNetLayer(numFilters, filterLength, Fs, numChannels, name);
layers(2) = sNL;

[sincNet, sincNetInfo] = trainNetwork(XTrain, TTrain, layers, options);

% frequency response of SincNet 
figure
plotNFilters(sincNet.Layers(2), 9);

% results summary
NetworkType = ["Standard CNN"; "SincNet Layer"];
Accuracy = [convNetInfo.FinalValidationAccuracy; sincNetInfo.FinalValidationAccuracy];

resultsSummary = table(NetworkType, Accuracy);

epoch = linspace(0, numEpochs, numel(sincNetInfo.ValidationAccuracy(~isnan(sincNetInfo.ValidationAccuracy))));
epoch = [epoch, numEpochs];

sinc_valAcc = [sincNetInfo.ValidationAccuracy(~isnan(sincNetInfo.ValidationAccuracy)), ...
    sincNetInfo.FinalValidationAccuracy];
conv_valAcc = [convNetInfo.ValidationAccuracy(~isnan(convNetInfo.ValidationAccuracy)), ...
    convNetInfo.FinalValidationAccuracy];

figure;
plot(epoch, sinc_valAcc, "-*", MarkerSize=4);
hold on;
plot(epoch, conv_valAcc, "-*", MarkerSize=4);
ylabel("Frame-Level Accuracy (Test Set)");
xlabel("Epoch");
xlim([0, numEpochs+0.3]);
title("Frame-Level Accuracy Versus Epoch");
legend("sincNet","conv2dLayer", Location="southeast");
grid on;

% supporting function
function xp = preprocessAudioData(x, frameLength, overlapLength, Fs)

speechIdx = detectSpeech(x, Fs);
xp = zeros(1, frameLength, 1, 0);

for ii = 1:size(speechIdx, 1)
    % Isolate speech segment
    audioChunk = x(speechIdx(ii, 1):speechIdx(ii, 2));

    % Split into 200 ms chunks
    audioChunk = buffer(audioChunk, frameLength, overlapLength);
    audioChunk = reshape(audioChunk, 1, frameLength, 1, size(audioChunk, 2));

    % Concatenate with existing audio
    xp = cat(4, xp, audioChunk);
end
end