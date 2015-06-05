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

