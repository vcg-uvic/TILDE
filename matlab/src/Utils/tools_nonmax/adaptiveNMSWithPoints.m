%% adaptiveNMSWithPoints.m --- 
% 
% Filename: adaptiveNMSWithPoints.m
% Description: 
% Author: Yannick Verdie, Kwang Moo Yi
% Maintainer: Yannick Verdie, Kwang Moo Yi
% Created: Tue Jun 16 17:16:07 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:16:11 2015 (+0200)
%           By: Kwang
%     Update #: 1
% URL: 
% Doc URL: 
% Keywords: 
% Compatibility: 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Commentary: 
% 
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Change Log:
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Copyright (C), EPFL Computer Vision Lab.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Code:


function [ I2,J2,S2 ] = adaptiveNMSWithPoints( I,J,S,nbPoints, thr_r )
%UNTITLED4 Summary of this function goes here
%   Detailed explanation goes here

if ~exist('thr_r', 'var')
    thr_r = 10*10;
end

I = reshape(I,[numel(I),1]);
J = reshape(J,[numel(J),1]);
S = reshape(S,[numel(S),1]);

%sort by score
[~, idx] = sort(S,'descend');
%idx = idx(S(idx) > 0);%only positive score
idx = idx(1:min(10*nbPoints,size(S,1)));
try
    refdataPts = [I(idx) J(idx)]';
catch
    display(size(S));
    display(size(I));
    display(size(J));
    error('size mismatch here');
end
%get the radius
numPts = length(idx);
radius = Inf(numPts, 1);

% thr_r = 10*10;
%try something else
isFree = true(1,numPts);
% isFree(1,1) = false;%biggest blocked
for i = 2 : numPts
    xi = I(idx(i));yi = J(idx(i));
    val = [xi yi]';
    dataPts = refdataPts(:,1:i-1);
    biggerFree = (isFree(:,1:i-1));
    sqDistToAll = sum((repmat(val,1,i-1) - dataPts).^2,1);    %dist squared from mean to all points still active
    if (any(sqDistToAll(biggerFree) < thr_r)) %1 or more better scored keypoint too close
        isFree(i) = 0;
    end
end
% isFree(1,1) = true;%biggest returned

I2 = I(idx(isFree));
J2 = J(idx(isFree));
S2 = S(idx(isFree));
I2 = I2((1:min(nbPoints,size(I2,1))));
J2 = J2((1:min(nbPoints,size(J2,1))));
S2 = S2((1:min(nbPoints,size(S2,1))));
% %sort the radius and throw away smaller than threshold
% [sorted_radius,idx2] = sort(radius, 'descend');
% idx3 = find(sorted_radius < 10*10);
% if numel(idx3) == 0
%     % do nothing if we find nothing from serching small ones
% else
%     I = I(idx(idx2(1:idx3(1)-1)));
%     J = J(idx(idx2(1:idx3(1)-1)));
%     S = S(idx(idx2(1:idx3(1)-1)));
% end
% 
% [~, idx4] = sort(S,'descend');
% idx4 = reshape(idx4, [numel(idx4),1]);
% 
% % I1 = I(idx(1:nbPoints));
% % J1 = J(idx(1:nbPoints));
% % [ binary_img1,score_img1 ] = pts2binary( I1,J1, score_img );
% % imshow(binary_img1);
% % 
% % figure;
% 
% I = I(idx4(1:min(nbPoints,size(idx4,1))));
% J = J(idx4(1:min(nbPoints,size(idx4,1))));
% J = J(idx(idx2(1:nbPoints)));

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% adaptiveNMSWithPoints.m ends here
