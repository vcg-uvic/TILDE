function v = cubicInterp3(V, X1, X2, X3)
% CUBICINTERP3 does a 3d cubic interpolation of V at (X1, X2, X3).
%
% See also: detectPoints, nonMax
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

%% function of 4 neighboring values
I = @(I) cat(4, ((2 - I) .* I - 1) ./ 2 .* I, ...
                (3 * I - 5) ./ 2 .* I.^2 + 1, ...
                ((4 - 3 * I) .* I + 1) ./ 2 .* I, ...
                (I - 1) ./ 2 .* I.^2);

%% indices of neighboring values
s0 = cumprod([1 size(V, 1), size(V, 2)]);
b = floor(mod([0 : 16 : 1023; 0 : 4 : 255; 0 : 63]' / 16, 4));
a = 1 + s0(1) * floor(X1 - 2) ...
      + s0(2) * floor(X2 - 2) ...
      + s0(3) * floor(X3 - 2);
B = reshape(b * s0', [1, 1, 1, 64]);
idx = a(:, :, :, ones(1, 64)) + B(ones(1, size(a, 1)), ...
                                  ones(1, size(a, 2)), ...
                                  ones(1, size(a, 3)), :);

%% fraction of coordinates
x1 = I(X1 - fix(X1));
x2 = I(X2 - fix(X2));
x3 = I(X3 - fix(X3));

%% interpolated value
v = sum(V(idx) .* x1(:, :, :, b(:, 1) + 1) ...
               .* x2(:, :, :, b(:, 2) + 1) ...
               .* x3(:, :, :, b(:, 3) + 1), 4);
