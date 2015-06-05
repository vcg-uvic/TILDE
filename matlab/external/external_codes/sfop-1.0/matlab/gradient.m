function g = gradient(C)
% GRADIENT computes the 3d gradient from a local neighborhood C.
%
% See also: detectPoints
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

%% gradient
d = 1 ./ [32, 16, 32;
          16,  8, 16;
          32, 16, 32];
d3 = cat(3, -d, zeros(3), d);
d1 = shiftdim(d3, 2);
d2 = shiftdim(d1, 2);
g = [d1(:), d2(:), d3(:)]' * C(:);
