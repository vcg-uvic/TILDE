function [r, c, s] = nonMax(p, r, c, s, params)
% NONMAX does a non-maximum suppression at (R, C, S) w. r. t. precision P.
%
% See also: detectPoints, cubicInterp3
%
% Licence:
%   For internal use only.
%
% Warranty:
%   No warranty for validity of this implementation.
%
% Authors:
%   Wolfgang Foerstner, Timo Dickscheid, Falko Schindler
%   Department of Photogrammetry
%   Institute of Geodesy and Geoinformation
%   University of Bonn
%   Bonn, Germany
%
% Contact person:
%   Falko Schindler (falko.schindler@uni-bonn.de)
%
% Copyright 2009-2011

%% write coordinates into table sorted by precision
table = [r, c, s, p];
[X, idx] = sort(table, 1, 'descend');
table = table(idx(:, 4), :);

%% process each point
line1 = 1;
s2sigma = @(s) params.sigma0 * 2.^((s - 3) / params.layersPerOctave);
while line1 < size(table, 1)
    
    % compare to following points (with lower precision)
    line2 = line1 + 1 : size(table, 1);
    
    % extract sigmas
    sig1 = s2sigma(table(line1, 3));
    sig2 = s2sigma(table(line2, 3));
    
    % Mahalanobis distance
    d2 =   (table(line1, 1) - table(line2, 1)).^2 ./ (sig1^2 + sig2.^2) ...
         + (table(line1, 2) - table(line2, 2)).^2 ./ (sig1^2 + sig2.^2) ...
         + (table(line1, 3) - table(line2, 3)).^2 ./ ...
         (params.nonmaxOctave * params.layersPerOctave)^2;
    
    % remove all points closer than threshold
    table(line2(d2 < params.nonmaxTd2), :) = [];
    
    % proceed with next point
    line1 = line1 + 1;
    
end

%% write points back into coordinate vectors
r = table(:, 1);
c = table(:, 2);
s = table(:, 3);
