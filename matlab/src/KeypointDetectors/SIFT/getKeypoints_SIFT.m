%% getKeypoints_SIFT.m --- 
% 
% Filename: getKeypoints_SIFT.m
% Description: Wrapper Function for SIFT
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:20:35 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Fri Aug 28 15:03:43 2015 (+0200)
%           By: Kwang
%     Update #: 6
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


function [keypts] = getKeypoints_SIFT(img_info, p)

    sift_name = [img_info.full_feature_prefix '_SIFT_keypoints.mat'];
    if ~exist(sift_name, 'file')

        in = img_info.image_name;
        in = strrep(in, 'image_gray', 'image_color');
        out = [img_info.full_feature_prefix '_SIFT_keypointstxt'];
        runKeypointsOpenCV('SIFT', in, out)

        [feat, ~, ~] = loadFeatures(out);
        feat = feat';
        score = load([out '.score']);

       % Get the scale 
        a = feat(3,:);
        b = feat(4,:);
        c = feat(5,:);
        % obtain scales
        scale = sqrt(a.*c - b.^2); % sqrt of determinant (sqrt of product of eigs)
        scale = 1./sqrt(scale); % inverse becuz it's actually inv of [a b; b c] and also
                                % additional sqrt as a = 1/scale^2 if b = 0

        keypts = [feat(1:2,:); zeros(5,size(feat,2))];
        keypts(5,:) = score';
        keypts(6,:) = scale';
        
        save(sift_name, 'keypts', '-v7.3');
    else
        loadkey = load(sift_name);
        keypts = loadkey.keypts;
    end
end


% function [keypts] = getKeypoints_SIFT(img_info, p)

%     sift_name = [img_info.full_feature_prefix '_sift_keypoints.mat'];
%     if ~exist(sift_name, 'file')
%         keypts = image2Sift(img_info.image_gray, p.peak_thresh,p.edge_thresh);
%         save(sift_name, 'keypts', '-v7.3');
%     else
%         loadkey = load(sift_name);
%         keypts = loadkey.keypts;
%     end

% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getKeypoints_SIFT.m ends here
