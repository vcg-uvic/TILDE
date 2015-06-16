%% getKeypoints_FAST.m --- 
% 
% Filename: getKeypoints_FAST.m
% Description: Wrapper Function for FAST
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:16:52 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:19:36 2015 (+0200)
%           By: Kwang
%     Update #: 2
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
%% Code:


% function [keypts] = getKeypoints_FAST(img_info, p)

%     fast_name = [img_info.full_feature_prefix '_FAST_keypoints.mat'];
%     if ~exist(fast_name, 'file')

%         in = img_info.image_name;
%         in = strrep(in, 'image_gray', 'image_color');
%         out = [img_info.full_feature_prefix '_FAST_keypointstxt'];
%         runKeypointsOpenCV('FAST', in, out)

%         [feat, ~, ~] = loadFeatures(out);
%         feat = feat';
%         score = load([out '.score']);

%        % Get the scale 
%         a = feat(3,:);
%         b = feat(4,:);
%         c = feat(5,:);
%         % obtain scales
%         scale = sqrt(a.*c - b.^2); % sqrt of determinant (sqrt of product of eigs)
%         scale = 1./scale; % inverse becuz it's actually inv of [a b; b c]

%         keypts = [feat(1:2,:); zeros(5,size(feat,2))];
%         keypts(5,:) = score';
%         keypts(6,:) = scale';
        
%         save(fast_name, 'keypts', '-v7.3');
%     else
%         loadkey = load(fast_name);
%         keypts = loadkey.keypts;
%     end
% end




function [keypts] = getKeypoints_FAST(img_info, p)
    fixed_scale = 7;%half of the filter size
    fast_name = [img_info.full_feature_prefix '_FAST9_keypoints.mat'];
    if ~exist(fast_name, 'file')
        [c9, scoresc9]= fast9(double(img_info.image_gray), 10,1);
        keypts = [c9'; zeros(3,size(c9,1)); repmat(fixed_scale,1,size(c9,1))];
        % convert score image to keypoint scores
        keypts = mergeScoreImg2Keypoints(keypts, scoresc9);
        save(fast_name, 'keypts', '-v7.3');
    else
        loadkey = load(fast_name);
        keypts = loadkey.keypts;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getKeypoints_FAST.m ends here
