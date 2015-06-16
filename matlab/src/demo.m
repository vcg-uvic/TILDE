%% demo.m --- 
% 
% Filename: demo.m
% Description: 
% Author: Yannick Verdie, Kwang Moo Yi
% Maintainer: Yannick Verdie, Kwang Moo Yi
% Created: Tue Jun 16 17:10:23 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:11:22 2015 (+0200)
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
% Copyright (C), EPFL Computer Vision Lab.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Code:


%%parameters..., no need to edit other than these 3 values%%
fullPathFilter = '../filters/BestFilters_2percents/Original/MexicoMed.mat';
fixed_scale = 10;
fullPathName = '../../data/testImage.png';
%%--------------------------------------------------------%%

%%setup the path....
addpath('Utils');
global sRoot;
tmp = mfilename('fullpath');tmp =  strsplit(tmp, '/');tmp= tmp(1:end-2);
sRoot = strjoin(tmp,'/');
setup_path;

%%setup the other parameters
Img = imread(fullPathName);

%%compute the keypoints and scores....
[ binary_res, score ] = ApplyLearnedELLFilter(Img, -inf, fullPathFilter, false );   
 idx = find(binary_res);
[I,J] = ind2sub(size(binary_res),idx);
features = [J I zeros(size(I,1),3) repmat(fixed_scale,size(I,1),1)]';
features = mergeScoreImg2Keypoints(features, score);

%sort by score
[~,idx] = sort(-features(5,:))
features = features(:,idx);
%keep the 500 best
features = features(:,1:min(size(features,2),500));

%display.....
figure;
imshow(Img);hold on;
plot(features(1,:), features(2,:), 'b.');
figure;
imshow((score - min(min(score)))/((max(max(score))-min(min(score)))))

fprintf('Program terminated normally.\n');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% demo.m ends here
