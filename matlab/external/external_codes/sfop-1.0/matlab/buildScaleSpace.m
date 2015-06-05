function [P, L2] = buildScaleSpace(img, params, scale)
% BUILDSCALESPACE computes the scale space for one image.
%
% Additional layers are added above and below the requested LAYERSPEROCTAVE
% layers. This is for later interpolation.
%
% See also: sfop, gaussFFT, omega
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

%% process each scale
[L2, P] = deal(zeros([size(img), params.layersPerOctave + 6]));
fprintf('Computing %dx%dx%d scale space of octave %d (%d/%d)\n', ...
    size(P), params.octave, params.octave + 1, params.numberOfOctaves);
for s = 1 : size(P, 3); fprintf('_'); end; fprintf('\n');
for s = 1 : size(P, 3)
    
    % skip layers for subpixel iterations if possible
    if params.koetheMaxIter <= 1 && (s < 3 || s > size(P, 3) - 2)
        fprintf('.');
        continue
    else
        fprintf('''');        
    end
    
    % scales for current layer wrt. current octave
    sig = params.sigma0 * ...
        2^((s - 4) / params.layersPerOctave) * 2^params.octave / scale;
    tau = params.sigma2tau(sig);
    
    % differentiation
    [gTau.r, gTau.c] = gaussFFT(img, tau, params.LUT.time, 'Gy', 'Gx');
    M = 12 * sig^2 + 1;
    
    % smallest eigenvalue of the structure tensor
    N0.rr = M * gaussFFT(gTau.r .* gTau.r, sig, params.LUT.time, 'G');
    N0.cc = M * gaussFFT(gTau.c .* gTau.c, sig, params.LUT.time, 'G');
    N0.rc = M * gaussFFT(gTau.r .* gTau.c, sig, params.LUT.time, 'G');
    tr  = N0.rr + N0.cc;
    det = N0.rr .* N0.cc - N0.rc.^2;
    L2(:, :, s) = real(0.5 * (tr - sqrt(tr.^2 - 4 * det)));

    % precision
    if strcmp(params.type, 'entropy')
        imgTau    = gaussFFT(img   , tau, params.LUT.time, 'G');
        imgTauSig = gaussFFT(imgTau, sig, params.LUT.time, 'G');
        Vg = gaussFFT((imgTau - imgTauSig).^2, sig, LUT.time, 'G');
        VnTau = params.noise^2 / 8 / pi / tau^4;
        P(:, :, s) = sqrt(L2(:, :, s) ./ Vg .* log(1.29 + Vg ./ VnTau));
    elseif strcmp(params.type, 'min')
        S0 = omega(  0, gTau, sig, M, params.LUT.time);
        S1 = omega( 60, gTau, sig, M, params.LUT.time);
        S2 = omega(120, gTau, sig, M, params.LUT.time);
        % model: S = a - b * cos(2 * alpha - 2 * alpha0)
        a = (S0 + S1 + S2) ./ 3;
        b = 2 / 3 * sqrt(  S0.^2    + S1.^2    + S2.^2 ...
                         - S0 .* S1 - S1 .* S2 - S2 .* S0);
        % alpha0 = 1 / 2 * atan2(sqrt(3) * (S2 - S1), S1 + S2 - 2 * S0);
        S = a - b;
        R = M - 3;
        P(:, :, s) = L2(:, :, s) .* R ./ S;
    elseif isnumeric(params.type)
        S = omega(params.type, gTau, sig, M, params.LUT.time);
        R = M - 2;
        P(:, :, s) = L2(:, :, s) .* R ./ S;
    else
        error('Invalid ''type''. Must be ''min'', ''entropy'' or numeric.')
    end
    
end
fprintf('\n');

%% correct for current scale
L2 = L2 / scale^2;
P = P * scale^2;

%% remove imaginary and undefined precisions
P = real(P);
P(isnan(P)) = -Inf;
