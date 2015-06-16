%% ApplyAdaptiveNonMax.m --- 
% 
% Filename: ApplyAdaptiveNonMax.m
% Description: 
% Author: Yannick Verdie, Kwang Moo Yi
% Maintainer: Yannick Verdie, Kwang Moo Yi
% Created: Tue Jun 16 17:15:42 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:15:46 2015 (+0200)
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


function [ binary_img ] = ApplyAdaptiveNonMax( score_img, binary_img, nbPoints )
%ApplyAdaptiveNonMax Apply Adaptive Nonmax
%   Detailed explanation goes here

if ~exist('nbPoints', 'var')
    nbPoints = 100;    
end

[ I,J,S ] = binary2pts( binary_img,score_img );

[I2, J2] = adaptiveNMSWithPoints( I,J,S,nbPoints );

[ binary_img,~ ] = pts2binary( I2,J2, score_img );
% imshow(binary_img2);

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ApplyAdaptiveNonMax.m ends here
