function showFeatures(imgFile, keyFiles, updatePlot, color, width, Width)
% SHOWFEATURES shows an image with detected keypoints in one plot.
%
% SHOWFEATURES(IMGFILE, KEYFILE) shows an image IMGFILE and reads an ASCII
% file DATFILE in the sense of
% http://www.robots.ox.ac.uk/~vgg/research/affine/detectors.html in order
% to display the keypoints as colored ellipses.
%
% Optional parameters are:
%   keyFiles     is assumed to be [imgFile '.sfop'] by default
%   updatePlot   if the plot should be redrawn or updated only
%   color        color of the ellipses
%   width        line width of the colored ellipses
%   Width        line width of the underlying black ellipses
%
% See also: loadFeatures
%
% Licence:
%   For internal use only.
%
% Author:
%   Falko Schindler (falko.schindler@uni-bonn.de)
%   Department of Photogrammetry
%   Institute of Geodesy and Geoinformation
%   University of Bonn
%   Bonn, Germany
%
% Example:
%   showFeatures('../examples/lena.png', '/tmp/lena.sfop');
%
% Copyright 2009-2011

%% read function parameters
if ~exist('keyFiles', 'var'), keyFiles = [imgFile '.sfop']; end
if ~exist('updatePlot', 'var'), updatePlot = false; end
if ~exist('color', 'var'), color = 'y'; end
if ~exist('width', 'var'), width = 2; end
if ~exist('Width', 'var'), Width = width + 3; end
if ischar(keyFiles), keyFiles = {keyFiles}; end;

%% read keypoint file
feat = loadFeatures(keyFiles);

%% prepare plot
if ~updatePlot
    clf;
    imshow(imread(imgFile));
end
hold on;

%% plot ellipses
t = 0 : pi / 50 : 2 * pi;
for c = 1 : size(feat, 1)
    Sigma = [feat(c, 3) feat(c, 4);
             feat(c, 4) feat(c, 5)];
    X = sqrtm(Sigma)^-1 * [cos(t); sin(t)];
    plot(X(1, :) + feat(c, 1) + 1, X(2, :) + feat(c, 2) + 1, ...
         '-', 'Color', 'k', 'LineWidth', Width);
end
for c = 1 : size(feat, 1)
    Sigma = [feat(c, 3) feat(c, 4);
             feat(c, 4) feat(c, 5)];
    X = sqrtm(Sigma)^-1 * [cos(t); sin(t)];
    plot(X(1, :) + feat(c, 1) + 1, X(2, :) + feat(c, 2) + 1, ...
         '-', 'Color', color, 'LineWidth', width);
end
