%% fastELLFiltering_approx.m --- 
% 
% Filename: fastELLFiltering_approx.m
% Description: 
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:15:26 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:15:30 2015 (+0200)
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


function [ score_res, computation_time ] = fastELLFiltering_approx(input_preproc_image, threshold, res)

    param = res.param;
    w_approx = res.newFilters;
    new_filters_per_channel = res.new_filters_per_channel;
    coeff_per_channel = res.coeff_per_channel;
    w_mean = res.w_mean;
% $$$     w_mean_correct = res.w_mean_correct;
    coeff = res.coeff;
    % un-normalize the coefficients
    %coeff = coeff .* repmat(reshape(w_mean,[1, numel(w_mean)]),[size(coeff,1), 1]);
    
    b_spatial = res.b_spatial;
    delta = res.delta;
    numChannel = size(input_preproc_image,3);
        
    tic;
    computation_time.ifft_time = toc;


    %out of tic toc because we can do that "offline" and save the results instead of the filters....
	
    if true
%         numChannel = length(new_filters_per_channel);
        numSepFilter = size(new_filters_per_channel{1},3);
        K1 = cell(numChannel, numSepFilter);
        K2 = cell(numChannel, numSepFilter);
        for idxChannel = 1:numChannel
            for idxFilter = 1:numSepFilter
                channelFilter = new_filters_per_channel{idxChannel};
                [U,S,V] = svd(channelFilter(:,:,idxFilter));
                K1{idxChannel, idxFilter} = U(:,1) * sqrt(S(1,1));
                K2{idxChannel, idxFilter} = V(:,1)' * sqrt(S(1,1));
            end
        end
    end

    b_spatial_repmat = repmat(shiftdim(b_spatial,-1), [size(input_preproc_image,1), size(input_preproc_image,2), 1]);

    numOrigFilter = size(coeff_per_channel{1},2);
    score_res_sep = zeros(numChannel,size(input_preproc_image,1), size(input_preproc_image,2), numSepFilter);
    score_res = zeros(size(input_preproc_image,1), size(input_preproc_image,2), numOrigFilter);
    score_res_sep_channel = zeros(size(input_preproc_image,1), size(input_preproc_image,2), numSepFilter);%, numChannel);

%     K1_reshape = zeros(size(K1{1},1), size(K1{1},2), numSepFilter*numChannel);
%     K2_reshape = zeros(size(K1{1},1), size(K1{1},2), numSepFilter*numChannel);
%     input_preproc_image_rep = zeros(size(input_preproc_image,1), size(input_preproc_image,2), numSepFilter*numChannel);
%     for idxChannel = 1:numChannel
%         for idxFilter = 1:numSepFilter
%             K1_reshape(:,:,idxFilter + (idxChannel-1)*numSepFilter) = K1{idxChannel,idxFilter};
%             K2_reshape(:,:,idxFilter + (idxChannel-1)*numSepFilter) = K2{idxChannel,idxFilter};
%             input_preproc_image_rep(:,:,idxFilter + (idxChannel-1)*numSepFilter) = input_preproc_image(:,:,idxChannel);
%         end
%     end

%     spmd
%         score_res_sep_channel = codistributed(score_res_sep_channel);
%         K1 = codistributed(K1);
%         K2 = codistributed(K2);
%     end

    tic;
    % apply each filter
    %     for idxFilteridxChannel = 1:numSepFilter*numChannel
    %         score_res_sep_channel(:,:,idxFilteridxChannel) = conv2(K1_reshape(:,:,idxFilteridxChannel),K2_reshape(:,:,idxFilteridxChannel),input_preproc_image_rep(:,:,idxFilteridxChannel),'symmetric');
    %     end
%     w_mean_correct = sum(w_mean_correct,2);
    %h,w,6,48
    for idxChannel = 1:numChannel
        co = coeff_per_channel{idxChannel}.*repmat(w_mean(idxChannel,:),[numSepFilter,1]);
        for idxFilter = 1:numSepFilter
%             score_res_sep_channel(:,:,idxFilter) = conv2(K1{idxChannel, idxFilter},K2{idxChannel, idxFilter},input_preproc_image(:,:,idxChannel),'symmetric');
            score_res_sep_channel(:,:,idxFilter) = imfilter(input_preproc_image(:,:,idxChannel),new_filters_per_channel{idxChannel}(:,:,idxFilter),'symmetric');
        end
        score_res_sep_reshape = reshape(score_res_sep_channel,[size(input_preproc_image,1)*size(input_preproc_image,2), size(co,1)]);
        score_res_reshape = score_res_sep_reshape * co;
% $$$         
% $$$         % TODO: Add the bias term induced from zero-mean condition
% $$$         % when performing tensor decomposition
% $$$         input_img_cur = input_preproc_image(:,:,idxChannel);
% $$$         score_res_reshape = score_res_reshape + input_img_cur(:); * w_mean_correct(idxChannel,:);
% $$$                                                           
        % Add the current separable to the final
        score_res(:) = score_res(:) + score_res_reshape(:);
    end
    score_res = score_res + b_spatial_repmat;
    
    computation_time.filtering_time = toc;
    
    tic;
    % do the max procedure
    score_hull = zeros(size(input_preproc_image,1), size(input_preproc_image,2), param.nDesiredHullNum);
    for idxHull = 1:param.nDesiredHullNum
        score_hull(:,:,idxHull) = delta(idxHull)*max(score_res(:,:,(idxHull-1)*param.nDesiredDepth+1:(idxHull)*param.nDesiredDepth),[],3);
    end

    % do the sum procedure
    score_res = sum(score_hull,3);

    % end measuring time
    computation_time.gathering_time = toc;


end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% fastELLFiltering_approx.m ends here
