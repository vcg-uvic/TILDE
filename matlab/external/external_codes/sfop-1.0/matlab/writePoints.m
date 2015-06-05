function writePoints(keyFile, r, c, radius)
% WRITEPOINTS writes detected keypoints to an ASCII file.
%
% See http://www.robots.ox.ac.uk/~vgg/research/affine/detectors.html for a
% desciption of such a keypoint file.
%
% See also: sfop
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

%% create directory
if ~isempty(fileparts(keyFile)) && ~exist(fileparts(keyFile), 'dir')
    mkdir(fileparts(datFile));
end

%% write file
f = fopen(keyFile, 'w');
fprintf(f, '1.0\n%d\n', numel(c));
fprintf(f, '%f %f %f %f %f\n', [[c, r] - 1, 1 ./ radius.^2 * [1, 0, 1]]');
fclose(f);
