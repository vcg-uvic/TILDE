function [ binary_img ] = ApplyHarris( score_img, binary_img )
%ApplyHarris : remove points from non-corner places
%   Detailed explanation goes here
warning('vl_harris disabled due to unknown bug');

%     [~ , h] = vl_harris(single(score_img),1,0); %h is 0<lambdamin/lambdamax<1
% %     imshow((1./h.rho)<5)%sift does 1<lambdamax/lambdamin<10(threshold at 10)
%     binary_img = binary_img .* ((1./h.rho)<5);

end

