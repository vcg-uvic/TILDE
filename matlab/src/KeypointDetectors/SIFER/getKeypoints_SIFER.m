%% getKeypoints_SIFER.m --- 
% 
% Filename: getKeypoints_SIFER.m
% Description: Wrapper Function for SIFER
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:20:14 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:20:29 2015 (+0200)
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


function [keypts] = getKeypoints_SIFER(img_info, p)

    sifer_name = [img_info.full_feature_prefix '_SIFER_keypoints.mat'];
    if ~exist(sifer_name, 'file')

        in = img_info.image_name;
        out = [img_info.full_feature_prefix '_SIFER_keypointstxt'];
        failed = sifer(in,out);
        
        if (failed)
            warning('SIFER is TURNED OFF (binary not found)');
            keypts = [];
            return;
        end

        bFeatureReady = false;
        numTry = 0;
        while ~bFeatureReady
            try
                [feat, ~, ~] = loadFeatures(out);
                feat = feat';
                score = load([out '.score']);
                bFeatureReady = true;
            catch
                warning('retrying...');
                pause(1);
                numTry = numTry + 1;
                if(numTry == 10)
                    error('max try!');
                end
            end
        end

        % Get the scale 
        a = feat(3,:);
        b = feat(4,:);
        c = feat(5,:);
        % obtain scales
        scale = sqrt(a.*c - b.^2); % sqrt of determinant (sqrt of product of eigs)
        scale = 1./sqrt(scale); % inverse becuz it's actually inv of [a b; b c]

        keypts = [feat(1:2,:); zeros(5,size(feat,2))];
        keypts(5,:) = score';
        keypts(6,:) = scale';
        
        save(sifer_name, 'keypts', '-v7.3');
    else
        loadkey = load(sifer_name);
        keypts = loadkey.keypts;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% getKeypoints_SIFER.m ends here
