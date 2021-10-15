function [lossGenerator, lossDiscriminator] = ganLoss(probReal, probGenerated)
%GANLOSS Compute the total loss of the GAN.

% Copyright 2020 The MathWorks, Inc.

% Calculate losses for discriminator network
lossGenerated = -mean(log(1 - probGenerated));

% multiplied probReal by 0.9 to implement
% one-sided labeling
lossReal = -mean(log(0.9*probReal));

% Combine losses for discriminator network
lossDiscriminator = lossReal + lossGenerated;

% Calculate loss for generator network
lossGenerator = -mean(log(probGenerated));
end