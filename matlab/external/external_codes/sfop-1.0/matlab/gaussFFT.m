function varargout = gaussFFT(img, sigma, varargin)
% GAUSSFFT convolves an image IMG with Gaussian kernels via FFT.
% 
% Kernels with SIGMA smaller than 0.4 pixels will be neglected since the
% algorithm gets very slow and thus not suited for small scales.
% Kernels with SIGMA smaller than 10 pixels will be convolved without FFT
% transform.
%
% Input:
%   img         image to be convolved
%   sigma       kernel parameter of length 1 or 2
%   varargin    first optional argument can be a look-up table for fft2
%               computational costs in order to predict optimal image
%               sizes (vector format),
%               all other arguments are kernel names, possible names are:
%                 'G',
%                 'Gx', 'Gy',
%                 'Gxx', 'Gyy', 'Gxy',
%                 'x2G', 'y2G', 'xyG'
%
% Output:
%   varargout   convolved images, corresponding to one kernel each
%
% Example:
%   img = zeros(512);
%   img(150, 100) = 1;
%   fftResult = gaussFFT(img, 20, 'G');
%   imshow(fftResult, []);
%
% See also: buildScaleSpace, omega
%
% Licence:
%   For internal use only.
%
% Warranty:
%   No warranty for validity of this implementation.
%
% Author:
%   Falko Schindler (falko.schindler@uni-bonn.de)
%   Department of Photogrammetry
%   Institute of Geodesy and Geoinformation
%   University of Bonn
%   Bonn, Germany
%
% Copyright 2009-2011

%% use standard convolution for small sigma
if sigma < 10
    varargout = cell(size(varargin));
    for a = 2 : numel(varargin)
        [filter_x, filter_y] = getFilters(sigma, varargin{a});
        varargout{a - 1} = imfilter(imfilter(img, filter_x, 'same'), filter_y, 'same');
    %     varargout{a - 1} = imfilter(img, getFilter(sigma, varargin{a}), 'same');
    end
    return
end

%% get two values for sigma not smaller than 0.4
% duplicate a single value
if numel(sigma) == 1
    sigma = [sigma, sigma];
end
% neglect small values
sigma = sigma .* (sigma >= 0.4);

%% pad and transform image
% image padding in spatial domain, depending on sigma
padding = 4;
% default size of padding
pad = round(sigma * padding);
% use LUT
if isnumeric(varargin{1})
    % use LUT for fft2 computational times
    time = varargin{1};
    varargin = varargin(2:end);
    % minimum size of padding
    padMin = round(sigma * padding);
    % maximum size of padding
    padMax = (2.^ceil(log2(2 * padMin + size(img))) - size(img)) ./ 2;
    % optimal size of padding
    pad = zeros(1, 2);
    for dim = 1 : 2
        % possible image sizes
        sizes = size(img, dim) + 2 * (padMin(dim) : padMax(dim));
        % fastest image size
        [t, idx] = min(time(sizes <= numel(time)));
        % optimal padding
        if ~isempty(idx), pad(dim) = (sizes(idx) - size(img, dim)) / 2; end
    end
end
% signal f: padded image
f = img([ones(1, pad(1)), 1 : size(img, 1), end * ones(1, pad(1))], ...
        [ones(1, pad(2)), 1 : size(img, 2), end * ones(1, pad(2))]);
% Fast Fourier Transformation of the image
F = fft2(f);
% size of padded image
N = size(F);

%% gaussians Hc for rows and columns separately
% transition function: gaussians
Hc = cell(1, 2);
% build them for rows and columns separately
for dim = 1 : 2
    % circular repetitions in frequency domain
    if sigma(dim) > 0
        nShah = ceil(6 / sigma(dim));
    else
        nShah = 0;
    end
    % multiple repetitions of each pixel
    [j, n] = ndgrid(-nShah : nShah, 0 : N(dim) - 1);
    % gaussian value at each position
    Hc{dim} = exp(-2 .* (sigma(dim) .* pi .* (j + n ./ N(dim))).^2);
end
% sum over all repetitions
Hy_temp = sum(Hc{1}, 1);
Hx_temp = sum(Hc{2}, 1);
% normalization (divide by N, so it won't have to be done when using fft2
% instead of ifft2)
Hy = Hy_temp ./ Hy_temp(1, 1) ./ N(1);
Hx = Hx_temp ./ Hx_temp(1, 1) ./ N(2);

%% construct kernels
% see handwritten notes
dy  = - i *      sin(2 * pi * (0 : N(1) - 1) ./ N(1));
dx  =   i *      sin(2 * pi * (0 : N(2) - 1) ./ N(2));
dyy = - 2 * (1 - cos(2 * pi * (0 : N(1) - 1) ./ N(1)));
dxx = - 2 * (1 - cos(2 * pi * (0 : N(2) - 1) ./ N(2)));
for k = 1 : length(varargin)
    switch varargin{k}
        case 'G',   H =  Hy'         *  Hx;
        case 'Gy',  H = (Hy .* dy)'  *  Hx;
        case 'Gx',  H =  Hy'         * (Hx .* dx);
        case 'Gyy', H = (Hy .* dyy)' *  Hx;
        case 'Gxx', H =  Hy'         * (Hx .* dxx);
        case 'Gxy', H = (Hy .* dy)'  * (Hx .* dx);
        case 'y2G', H = (Hy .* dyy)' *  Hx         * sigma(1)^4 + ...
                         Hy'         *  Hx         * sigma(1)^2;
        case 'x2G', H =  Hy'         * (Hx .* dxx) * sigma(2)^4 + ...
                         Hy'         *  Hx         * sigma(2)^2;
        case 'xyG', H = (Hy .* dy)'  * (Hx .* dx)  * prod(sigma)^2;
        otherwise,  error 'gaussFFT: unknown kernel name';
    end
    % fft convolution (inverse fft transformation via ifft, normalization
    % is already done)
    G = real(fft2((F .* H)')');
    % crop result
    varargout{k} = G(pad(1) + 1 : end - pad(1), pad(2) + 1 : end - pad(2));
end
