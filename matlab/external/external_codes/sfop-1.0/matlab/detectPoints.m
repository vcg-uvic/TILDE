function [r, c, s, p] = detectPoints(P, L2, params, scale)
% DETECTPOINTS extracts interest points from the scale space.
%
% See also: sfop, cubicInterp3, hessian, gradient, nonMax
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

%% local maxima
[r, c, s] = ind2sub(size(P), find(imregionalmax(P)));
fprintf('Found %d local maxima, ', numel(r));

%% remove points close to the boundary (wrt. local scale)
sig = params.sigma0 * ...
    2.^((s - 4) / params.layersPerOctave) * 2^params.octave / scale;
border = max(2, 2.5 * sig);
valid = r - 1 >= border & size(P, 1) - r >= border & ...
        c - 1 >= border & size(P, 2) - c >= border & ...
        s - 1 >= 3      & size(P, 3) - s >= 3;
[r, c, s, sig] = deal(r(valid), c(valid), s(valid), sig(valid));
fprintf('%d off boundary, ', numel(r));

%% threshold on p and on lambda (wrt. global scale)
Sig = sig * scale;
h = params.noise^2 ./ 16 ./ pi ./ params.sigma2tau(Sig).^4;
Tl2 = h .* params.Tlambda2 .* chi2inv(0.999, 24 * Sig.^2 + 2);
valid = cubicInterp3(P, r, c, s) > params.Tp & ...
       cubicInterp3(L2, r, c, s) > Tl2; % NOTE: maybe not really necessary
%         L2(sub2ind(size(L2), round(r), round(c), round(s))) > Tl2;
[r, c, s] = deal(r(valid), c(valid), s(valid));
fprintf('%d above thresholds, ', numel(r));

%% optimize each point
[R, C, S] = ndgrid(-1 : 1);
valid = true(1, numel(r));
invalid = struct('H', 0, 'B', 0, 'I', 0);
for p = 1 : numel(r)
    % iterate multiple times
    for iter = 1 : params.koetheMaxIter
        % gradient and Hessian in current 27-neighborhood
        if iter == 1
            P27 = P(r(p) + (-1 : 1), c(p) + (-1 : 1), s(p) + (-1 : 1));
        else
            P27 = cubicInterp3(P, r(p) + R, c(p) + C, s(p) + S);
        end
        H = hessian(P27);
        if any(isnan(H(:))) || any(eig(H) > 0)
            valid(p) = false; invalid.H = invalid.H + 1; break; end
        % update position, but check for image boundary
        update = -H \ gradient(P27);
        r(p) = r(p) + update(1);
        c(p) = c(p) + update(2);
        s(p) = s(p) + update(3);
        if min([[r(p), c(p), s(p)] - 1, size(P) - [r(p), c(p), s(p)]]) < 2
            valid(p) = false; invalid.B = invalid.B + 1; break; end
        % check convergence
        if norm(update) < params.koetheEpsilon, break; end
        if iter == params.koetheMaxIter, valid(p) = false; invalid.I = invalid.I + 1; break; end
    end
end
[r, c, s] = deal(r(valid), c(valid), s(valid));
fprintf('%d converged.\n', numel(r));
fprintf('Invalid: %dx Hessian, %dx border, %dx #iterations\n', ...
    invalid.H, invalid.B, invalid.I);

%% return interpolated precision
p = cubicInterp3(P, r, c, s);

%% return global coordinates
r = (r - 1) * scale + 1;
c = (c - 1) * scale + 1;
s = s + params.layersPerOctave * params.octave;
