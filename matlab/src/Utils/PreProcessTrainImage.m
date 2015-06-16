%% PreProcessTrainImage.m --- 
% 
% Filename: PreProcessTrainImage.m
% Description: 
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:12:53 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:12:58 2015 (+0200)
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


function [ output_img ] = PreProcessTrainImage( input_img, param )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if isfield(param, 'fMultiScaleList')
    fMultiScaleList = param.fMultiScaleList;
    error('MultiScale is not Supported by This Function, should be treated before in some way!');
end

%% Check if AC is performed
bDoAC = false;
listDataFormat = strsplit(param.DataFormat, '_');
if strcmpi(listDataFormat{end}, 'ac')
    param.DataFormat = param.DataFormat(1:end-3);
    bDoAC = true;
end

%% Obtain the Multi Channel Image

output_img = PreProcessTrainImageSingleScale( input_img, param );

%% Run AC filters to augment AC dimention

if(bDoAC)
% Prepare blank initial context
    output_img = cat(3,output_img,ones(size(output_img,1),size(output_img,2)));

    % If there are ACFilters...
    if isfield(param, 'ACFilter_list')
        % read the list of learned filters 
        for idxAC = 1:length(param.ACFilter_list)
            curACFilter = param.ACFilter_list{idxAC};

            % apply them to the image
            [ ~, cur_res ] = ApplyLearnedFilterWithSVM_NoLoadFile( input_img, -inf, curACFilter, param, output_img );
            % do pooling
            ACImg = performPooling(cur_res,param);

            % swap with the new context
            output_img(:,:, end) = ACImg;
        end
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% PreProcessTrainImage.m ends here
