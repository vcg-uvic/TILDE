%% fastELLFiltering.m --- 
% 
% Filename: fastELLFiltering.m
% Description: 
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:15:18 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:15:22 2015 (+0200)
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


function [ score_res, computation_time, debug, b_spatial ] = fastELLFiltering(input_preproc_image, threshold, res)

	param = res.param;
	w_fourier_filters = res.w;
	delta = res.delta;
	nDim = size(input_preproc_image,3);

	tic;
	% convert to spatial domain filters
	w_spatial = zeros(param.nBinSize,param.nBinSize,nDim,size(w_fourier_filters,1));
	b_spatial = zeros(1,size(w_fourier_filters,1));
	for idxFilter = 1:size(res.w,1)
		[w_spatial(:,:,:,idxFilter), b_spatial(:,idxFilter)] = fourier2Spatial(w_fourier_filters(idxFilter,:), param);
	end
	computation_time.ifft_time = toc;

	tic
	% apply each filter
	score_res_all = zeros(size(input_preproc_image,1), size(input_preproc_image,2), size(w_spatial,4));
    score_res_all2 = zeros(size(input_preproc_image,1), size(input_preproc_image,2), size(w_spatial,4));
	for idxFilter = 1:size(res.w,1)
		score_res_tmp = zeros(size(input_preproc_image,1), size(input_preproc_image,2), nDim);
		for ii = 1:nDim
		    score_res_tmp(:,:,ii) = imfilter(double(input_preproc_image(:,:,ii)),w_spatial(:,:,ii,idxFilter), 'symmetric');
		end
		score_res_all(:,:,idxFilter) = sum(score_res_tmp,3) + b_spatial(1,idxFilter);
        score_res_all2(:,:,idxFilter) = sum(score_res_tmp,3);
	end
	computation_time.filtering_time = toc;
    debug = score_res_all2;

	tic
	% do the max procedure
	score_hull = zeros(size(input_preproc_image,1), size(input_preproc_image,2), param.nDesiredHullNum);
	for idxHull = 1:param.nDesiredHullNum
	    score_hull(:,:,idxHull) = delta(idxHull)*max(score_res_all(:,:,(idxHull-1)*param.nDesiredDepth+1:(idxHull)*param.nDesiredDepth),[],3);
	end

	% do the sum procedure
	score_res = sum(score_hull,3);

	% end measuring time
	computation_time.gathering_time = toc;


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% fastELLFiltering.m ends here
