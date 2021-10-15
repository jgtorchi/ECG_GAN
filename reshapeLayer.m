classdef reshapeLayer < nnet.layer.Layer
    properties
        OutputSize
    end
    properties (Learnable)
    end
    methods
        function layer = reshapeLayer(outputSize,name)
            layer.Name = name;
            % Set layer description
            layer.Description = "Reshape layer with output size " + join(string(outputSize));
            
            % Set layer type
            layer.Type = "Reshape";
            
            % Set output size
            layer.OutputSize = outputSize;
        end
        function [Z] = predict(layer, X)
            % Reshape
            outputSize = layer.OutputSize;
            Z = reshape(X,outputSize(1), outputSize(2), outputSize(3),[]);
        end
    end
    
end