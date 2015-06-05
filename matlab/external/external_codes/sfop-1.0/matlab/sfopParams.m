function params = sfopParams(varargin)
% SFOPPARAMS generates a struct with control parameters for SFOP detector.
%
% The default values can be changed via optional arguments VARARGIN like
%   sfopParams('Parameter', value);
% Call sfopParams for a list of possible parameter names and default
% values.
%
% See also: sfop
%
% Licence:
%   For internal use only.
%
% Authors:
%   Timo Dickscheid, Falko Schindler, Wolfgang Foerstner
%   Department of Photogrammetry
%   Institute of Geodesy and Geoinformation
%   University of Bonn
%   Bonn, Germany
%
% Contact person:
%   Falko Schindler (falko.schindler@uni-bonn.de)
%
% Copyright 2009-2011

%% default parameters
params.usePyramid = true;
params.layersPerOctave = 4;
params.numberOfOctaves = 3;
params.sigma0 = 2^(1 + 1 / params.layersPerOctave);
params.sigma2tau = @(sigma) sigma ./ 3;
params.Tp = -Inf;
params.Tlambda2 = 2;
params.maxNumFeatures = Inf;
params.noise = 0.02;
params.type = 'min';
params.nonmaxTd2 = 1;
params.nonmaxOctave = 0.5;
params.sizeFactor = 1;
params.koetheMaxIter = 1;
if params.koetheMaxIter <= 1
    params.koetheEpsilon = 1;
else
    params.koetheEpsilon = 0.2;
end

%% modifications via optional parameters
for i = 1 : 2 : numel(varargin) - 1
    if ismember(varargin{i}, fieldnames(params))
        params.(varargin{i}) = varargin{i + 1};
    else
        error('Unknown parameter ''%s''.', varargin{i});
    end
end
