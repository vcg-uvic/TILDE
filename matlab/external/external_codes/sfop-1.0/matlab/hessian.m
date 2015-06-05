function H = hessian(C)
% HESSIAN computes the 3d hessian from a local neighborhood C.
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

%% Hessian
d = 1 ./ [16, 8, 16;
           8, 4,  8;
          16, 8, 16];
d33 = cat(3, d, -2 * d, d);
d11 = shiftdim(d33, 2);
d22 = shiftdim(d11, 2);
d = 1 ./ [-16,  -8, -16;
          Inf, Inf, Inf;
           16,   8,  16];
d13 = cat(3, -d, zeros(3), d);
d12 = shiftdim(d13, 2);
d23 = shiftdim(d12, 2);
H = reshape([d11(:), d12(:), d13(:), ...
             d12(:), d22(:), d23(:), ...
             d13(:), d23(:), d33(:)]' * C(:), 3, 3);
