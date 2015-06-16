%% ApplyLearnedELLFilterWithPreProcImage_NoLoadFile.m --- 
% 
% Filename: ApplyLearnedELLFilterWithPreProcImage_NoLoadFile.m
% Description: 
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:12:31 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:12:35 2015 (+0200)
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


function [ binary_res, score_res_final ] = ApplyLearnedELLFilterWithPreProcImage_NoLoadFile( input_preproc_image, threshold, res, bDisp )
% %% Load Filter
% param = 0;
% load(nameELLFilter);
% param = rmfield(param, 'fMultiScaleList');

%% Load Filter
% load(nameDualCascade);

w = res.w;
delta = res.delta;
param = res.param;
if isfield(param, 'fMultiScaleList')
	param = rmfield(param, 'fMultiScaleList');
end

% do all filtering
score_res = zeros(size(input_preproc_image,1),size(input_preproc_image,2),size(w,1));
for idxW = 1:size(w,1)
% parfor idxW = 1:size(w,1)
    cur_w = w(idxW,:);
    [ ~, score_res(:,:,idxW) ] = ApplyLearnedFilterWithSVM_NoLoadFile( [], threshold, cur_w, param, input_preproc_image );
	if(sum(sum(isnan(score_res(:,:,idxW)))) + sum(sum(isinf(score_res(:,:,idxW)))) > 0)
	    warning('there are nan and inf in the filter responses!');
	    score_res(:,:,idxW) = zeros(size(score_res(:,:,idxW)));
	end
end


% max over the planes and apply delta
score_hull = zeros(size(input_preproc_image,1),size(input_preproc_image,2),param.nDesiredHullNum);
for idxHull = 1:param.nDesiredHullNum
% parfor idxHull = 1:param.nDesiredHullNum
    score_hull(:,:,idxHull) = delta(idxHull)*max(score_res(:,:,(idxHull-1)*param.nDesiredDepth+1:(idxHull)*param.nDesiredDepth),[],3);
end

score_res_final = sum(score_hull,3);

% % min over the planes in each cascade
% score_cascade = zeros(size(input_color_image,1),size(input_color_image,2),param.nDesiredHullNum);
% parfor idxEnsemble = 1:param.nDesiredHullNum
%     score_cascade(:,:,idxEnsemble) = min(score_res(:,:,(idxEnsemble-1)*param.nDesiredDepth+1:(idxEnsemble)*param.nDesiredDepth),[],3);
% end
% % max over the hulls
% score_res_final = max(score_cascade,[],3);

fs = param.nBinSize;
[score_res_final, max_img] = ApplyNonMax2Score(score_res_final, param);
% max_img = ApplyHarris(score_res_final, max_img);
% max_img = ApplyAdaptiveNonMax( score_res_final, max_img, nbPoints );
% warning('TODO: ApplyHarris, ApplyAdaptiveNonMax');

% cascade_safety = 1.0;
% if isfield(param, 'ACFilter_list')
%     cascade_safety = double(length(param.ACFilter_list));
% end
binary_res = max_img .* (score_res_final > threshold);
% Mutiplied fs with param.fScaling to consider scaling (25/04/2014 KMYI)
binary_res(1:fs,:) = 0;
binary_res(end-fs+1:end,:) = 0;
binary_res(:,1:fs) = 0;
binary_res(:,end-fs+1:end) = 0;


end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ApplyLearnedELLFilterWithPreProcImage_NoLoadFile.m ends here
