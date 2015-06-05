function sfop(imgFile, keyFile, varargin)
% SFOP detects SFOP-points in an image and writes them into an ASCII file.
%
% SFOP(IMGFILE, KEYFILE, VARARGIN) detects SFOP-keypoints in the image
% IMGFILE and writes them into the ASCII file DATFILE in the sense of
% http://www.robots.ox.ac.uk/~vgg/research/affine/detectors.html.
%
% Optional parameters can be passed as arguments like
%   sfop(imgFile, keyFile, 'Parameter1', value1, 'Parameter2', value2);
% Call sfopParams for a list of possible parameter names and default
% values.
% KEYFILE is optional: KEYFILE = [IMGFILE '.sfop'] is used by default.
%
% For further information refere to
%   http://www.ipb.uni-bonn.de/sfop/
% and cite as
%   @inproceedings{
%     foerstner*09:detecting,
%     author={W. F\"orstner and T. Dickscheid and F. Schindler},
%     title={Detecting Interpretable and Accurate Scale-Invariant
%            Keypoints},
%     booktitle={12th IEEE International Conference on Computer Vision
%                (ICCV'09)},
%     address={Kyoto, Japan},
%     year={2009}
%   }
%
% See also: sfopParams, setScales, buildScaleSpace, detectPoints,
%   writePoints
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
% Example:
%   sfop('../examples/lena.png', '/tmp/lena.sfop');
%   showFeatures('../examples/lena.png', '/tmp/lena.sfop');
%
% Contact person:
%   Falko Schindler (falko.schindler@uni-bonn.de)
%
% Copyright 2009-2011

%% parse parameters
if ~exist('keyFile', 'var'), keyFile = [imgFile '.sfop']; end
if numel(varargin) == 1, params = varargin;
else params = sfopParams(varargin{:}); end

%% load and normalize image to the range of 0..1
img = mean(im2double(imread(imgFile)), 3);

%% detect points on multiple octaves
[r, c, s, p] = deal([]);
params.LUT = load('fftLUT.mat');
scale = 1;
for octave = 0 : params.numberOfOctaves - 1

    %% rescale image
    params.octave = octave;
    if params.usePyramid && params.octave > 0
        img = gaussFFT(img, sqrt(0.5), params.LUT.time, 'G');
        img = img(1 : 2 : end, 1 : 2 : end);
        scale = scale * 2;
    end

    %% scale space
    [P, L2] = buildScaleSpace(img, params, scale);

    %% point detection
    [ri, ci, si, pi] = detectPoints(P, L2, params, scale);
    [r, c, s, p] = deal([r; ri], [c; ci], [s; si], [p; pi]); %#ok<AGROW>

end

%% supress non-maxima, note: points are sorted wrt. p after calling nonMax
[r, c, s] = nonMax(p, r, c, s, params);
fprintf('%d unique and ', numel(r));

%% constrain to best points only
best = 1 : min(params.maxNumFeatures, numel(r));
score = numel(r):-1:1;
score_file = fopen([keyFile '.score'], 'w');
for ii = 1:length(score)
    fprintf(score_file, '%f\n', score(ii));
end
fclose(score_file);
[r, c, s] = deal(r(best), c(best), s(best));
fprintf('%d best.\n', numel(r));

%% write point list
Sig = params.sigma0 * 2.^((s - 4) / params.layersPerOctave);
writePoints(keyFile, r, c, Sig * params.sizeFactor);
