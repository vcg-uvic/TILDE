function S = omega(alphaDeg, g, sig, M, fftTime)
% OMEGA computes the model error for a certain ALPHA on a gradient image G.
%
% See also: gaussFFT, buildScaleSpace
%
% Licence:
%   For internal use only.
%
% Warranty:
%   No warranty for validity of this implementation.
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

%% convert angle
alpha = alphaDeg / 180 * pi;

%% rotate gradients
gAlpha.r =   cos(alpha) .* g.r + sin(alpha) .* g.c;
gAlpha.c = - sin(alpha) .* g.r + cos(alpha) .* g.c;

%% compute error messure
S = M * (gaussFFT(gAlpha.r .* gAlpha.r, sig, fftTime, 'y2G') ...
   + 2 * gaussFFT(gAlpha.r .* gAlpha.c, sig, fftTime, 'xyG') ...
   +     gaussFFT(gAlpha.c .* gAlpha.c, sig, fftTime, 'x2G'));
