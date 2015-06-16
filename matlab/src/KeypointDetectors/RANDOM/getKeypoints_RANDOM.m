%% getKeypoints_RANDOM.m --- 
% 
% Filename: getKeypoints_RANDOM.m
% Description: 
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:18:28 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:18:33 2015 (+0200)
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


function [keypts] = getKeypoints_RANDOM(img_info, p)

    fixed_scale = 1;%half of the filter size

    % load image
    curImg = img_info.image_color;

    % binary keypoint image
    binary_res = ones(size(curImg,1),size(curImg,2));
    binary_res(randperm(numel(binary_res),3000)) = 1; % max 3000 for efficiency in sorting later
    score_res = rand(size(binary_res));

    % turn to keypts
    idx = find(binary_res);
    [I,J] = ind2sub(size(binary_res),idx);
    keypts = [J I zeros(size(I,1),3) repmat(fixed_scale,size(I,1),1)]';
    keypts = mergeScoreImg2Keypoints(keypts, score_res);

end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getKeypoints_RANDOM.m ends here
