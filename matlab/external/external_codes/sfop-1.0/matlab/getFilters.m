function [filter_x, filter_y] = getFilters(sigma, kernel)
%  getFilters  returns the filters used by gaussFFT.
%
%  Input:
%     sigma     = kernel parameter 1x1
%     kernel    = kernel identifier, possible names are:
%                     'G',
%                     'Gx', 'Gy',
%                     'Gxx', 'Gyy', 'Gxy',
%                     'xG', 'yG',
%                     'x2G', 'y2G', 'xyG'
%
%  Output:
%     filter_x  = kernel in image space of size (2*4*sigma)
%     filter_y  = kernel in image space of size (2*4*sigma)

% Example:
%     bar(getFilters(3, 'Gx'));

% See also fspecial, diff, gaussFFT
% Copyright 2009-2011 Falko Schindler (mail@falkoschindler.de)

%% construct kernels
kSize = floor(4 * sigma);
G_y = fspecial('gaussian', [2 * kSize + 1, 1], sigma);
G_x = fspecial('gaussian', [1, 2 * kSize + 1], sigma);
y = (-kSize : kSize)';
x =  -kSize : kSize;
dy = [-0.5, 0, 0.5]';
dx = [-0.5, 0, 0.5];
dyy = [0.25, -0.5, 0.25]';
dxx = [0.25, -0.5, 0.25];
switch kernel
    case 'G',   filter_x = G_x;
                filter_y = G_y;
    case 'Gy',  filter_x = G_x;
                filter_y = imfilter(G_y, dy, 'same', 'corr');
    case 'Gx',  filter_x = imfilter(G_x, dx, 'same', 'corr');
                filter_y = G_y;
    case 'Gyy', filter_x = G_x;
                filter_y = imfilter(G_y, dyy, 'same', 'corr');
    case 'Gxx', filter_x = imfilter(G_x, dxx, 'same', 'corr');
                filter_y = G_y;
    case 'Gxy', filter_x = imfilter(G_x, dx, 'same', 'corr');
                filter_y = imfilter(G_y, dy, 'same', 'corr');
    case 'yG',  filter_x = G_x;
                filter_y = y .* G_y;
    case 'xG',  filter_x = x .* G_x;
                filter_y = G_y;
    case 'y2G', filter_x = G_x;
                filter_y = y.^2 .* G_y;
    case 'x2G', filter_x = x.^2 .* G_x;
                filter_y = G_y;
    case 'xyG', filter_x = x .* G_x;
                filter_y = y .* G_y;
    otherwise,  error 'gaussFFT: unknown kernel name';
end
