classdef fullyConnected3D < nnet.layer.Layer
    properties
        OutputSize
    end
    properties (Learnable)
        % Layer learnable parameters
        
        Weights
        Bias
    end
    methods
        function layer = fullyConnected3D(outputSize,name)
            layer.Name = name;
            % Set layer description
            layer.Description = "Fully connected layer with output size " + join(string(outputSize));
            
            % Set layer type
            layer.Type = "Fully Connected 3D";
            
            % Set output size
            layer.OutputSize = outputSize;
            
            % Initialize fully connect weights and bias
            fcSize = prod(outputSize);
            layer.Weights = initializeGlorot(fcSize, numChannels);
            layer.Bias = zeros(fcSize, 1, 'single');
        end
        function [Z] = predict(layer, X)
            % Fully connect
            weights = layer.Weights;
            bias = layer.Bias;
            X = fullyconnect(X, weights, bias, 'DataFormat', 'SSCB');
            
            % Reshape
            outputSize = layer.OutputSize;
            Z = reshape(X,outputSize(1), outputSize(2), outputSize(3),[]);
        end
    end
    
end

function weights = initializeGlorot(numOut, numIn)
% Initialize weights using Glorot initializer.

varWeights = sqrt( 6 / (numIn + numOut) );
weights = varWeights * (2 * rand([numOut, numIn], 'single') - 1);

end