%% ApplyLearnedFilterWithSVM_NoLoadFile.m --- 
% 
% Filename: ApplyLearnedFilterWithSVM_NoLoadFile.m
% Description: 
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:12:41 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:12:46 2015 (+0200)
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


function [ binary_res, score_res ] = ApplyLearnedFilterWithSVM_NoLoadFile( input_color_image, threshold, trained_svmWeights, param, input_preproc_image )

if (~exist('param','var'))
    param.nFeatureType = 1;
    param.bDoSmooth = false;
    param.nameSVM = 'LearnedSVM_hardNegativeMine_mmfc_th';
    param.bUseSpatialDomain = false;
    param.fScaling = 1.0;
    param.nBinSize = 16;
end
param.bDoSmooth = false;
% param.bDoSmooth = true;

if (~exist('param.bOldFFT', 'var'))
    param.bOldFFT = false;
end

if ~isfield(param,'bDoFFT')
    param.bDoFFT = true;
end


%% Convert Filter into Spatial Domain

if param.bDoFFT && ~param.bUseSpatialDomain
    nDim = (length(trained_svmWeights)-1) / param.nBinSize / param.nBinSize / 2;
else
    nDim = (length(trained_svmWeights)-1) / param.nBinSize / param.nBinSize ;
end

if(param.bUseSpatialDomain || ~param.bDoFFT)
    learned_filter  = trained_svmWeights(1:end-1);

    size_filter = size(learned_filter,2);
    filter_orig_size = sqrt(size_filter/nDim);

    filter_img = reshape(learned_filter, [filter_orig_size filter_orig_size nDim]);

else
    learned_filter  = trained_svmWeights(1:end-1);

    size_filter = size(learned_filter,2)/2;

    clear i;%imaginary number, make sure it is not assigned before

    complex_filter_real = learned_filter(1:size_filter);
    complex_filter_imag = learned_filter(size_filter+1:size_filter*2);
    complex_filter = complex_filter_real + i*complex_filter_imag;

    filter_orig_size = sqrt(size_filter/nDim);

    complex_filter = reshape(complex_filter, [filter_orig_size filter_orig_size nDim]);
    complex_filter_ifft = zeros(filter_orig_size,filter_orig_size,nDim);
    for ii=1:nDim
        if(param.bOldFFT)
            complex_filter_ifft(:,:,ii) = ifft2(complex_filter(:,:,ii));
        else
            complex_filter_ifft(:,:,ii) = fftshift(ifft2(ifftshift(complex_filter(:,:,ii))));
        end
    end

    % complex_filter = reshape(trained_svmWeights(1:end-1), [filter_orig_size filter_orig_size 3]);
    filter_img = real(complex_filter_ifft);    
end

% filter_img = filter_img - ones(size(filter_img)) / (param.nBinSize*param.nBinSize);
%% Load Multichannel Image

if ~exist('input_preproc_image', 'var')
    if(size(input_color_image,3) == 1)
        input_color_image = repmat(input_color_image, [1 1 3]);
    end
    input_color_image = PreProcessTrainImage(input_color_image, param);
else
    % for AC compatibility
    input_color_image = input_preproc_image;
end

[orig_m, orig_n, orig_p] = size(input_color_image);

%% Re-warp the filters in their original shapes
warped_filter_list = cell(1,nDim);
for idxDim = 1:nDim
    cur_map = ones(param.nBinSize);

    tmp_filter = zeros(size(cur_map));
    tmp_filter(find(cur_map)) = filter_img(:,:,idxDim); 
    warped_filter_list{idxDim} = tmp_filter;
end

fs = param.nBinSize;

score_res_tmp = zeros(size(input_color_image));
for ii = 1:nDim
    score_res_tmp(:,:,ii) = imfilter(double(input_color_image(:,:,ii)),warped_filter_list{ii}, 'symmetric');
end
score_res_tmp = sum(score_res_tmp,3);
if param.bDoFFT
    score_res = (fs*fs) * score_res_tmp + real(trained_svmWeights(end));
else
    score_res = score_res_tmp + real(trained_svmWeights(end));
end


if(sum(sum(isnan(score_res))) + sum(sum(isinf(score_res))) > 0)
    warning('there are nan and inf in the filter responses!');
    score_res = zeros(size(score_res));
end

[score_res_fixed, max_img] = ApplyNonMax2Score(score_res, param);

cascade_safety = 1.0;

binary_res = max_img .* (score_res_fixed > threshold);
binary_res(1:fs*param.fScaling*cascade_safety,:) = 0;
binary_res(end-fs*param.fScaling*cascade_safety+1:end,:) = 0;
binary_res(:,1:fs*param.fScaling*cascade_safety) = 0;
binary_res(:,end-fs*param.fScaling*cascade_safety+1:end) = 0;

end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% ApplyLearnedFilterWithSVM_NoLoadFile.m ends here
