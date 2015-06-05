%% PreProcessTrainImageSingleScale.m --- 
% 
% Filename: PreProcessTrainImageSingleScale.m
% Description: Function which takes care of multiple channel composition
% Author: Kwang
% Maintainer: 
% Created: Thu Jan 15 10:33:30 2015 (+0100)
% Version: 
% Package-Requires: Separable filters, Dolar Toolbox, mexUtils,
%                   Descriptor fields
% Last-Updated: Fri Feb  6 16:38:43 2015 (+0100)
%           By: Kwang
%     Update #: 16
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


function [ output_img ] = PreProcessTrainImageSingleScale( input_img, param )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

%% Input Check
    if (~isfield(param,'DataFormat'))
        warning('Using with old mat...defaulting to RGB');
        param.DataFormat = 'rgb';
    end

    if(size(input_img,1) <= 32)
        % This function is being applied to a patch which is not intended
        error('Wrong Usage of this Function! This function is being applied to a patch which is not intended.');
    end

    %% Parameters
    % Parameters from Amos fo rthe separable filters
    % channel 1 filters (L)
    %learned
    fb{1,1} = {'separablefilters/mexico/fb_Mexico_L_36_21_sep_cpd_rank_20_rot_n1.txt';... %sep filter
               'separablefilters/mexico/fb_Mexico_L_36_21_weigths_cpd_rank_20_rot_n1.txt'}  ;%weigths
    select_idxs{1,1} = []; % to select subset of filters indicate indeces here, if empty use all filters
    fb{1,2} = {'separablefilters/mexico/fb_Mexico_L_36_11_sep_cpd_rank_20_rot_n1.txt';...
               'separablefilters/mexico/fb_Mexico_L_36_11_weigths_cpd_rank_20_rot_n1.txt'};
    select_idxs{1,2} = []; 
    % channel 2 filters (u)
    %learned
    fb{2,1} =  {'separablefilters/mexico/fb_Mexico_uv_36_21_sep_cpd_rank_20_rot_n1.txt';...
                'separablefilters/mexico/fb_Mexico_uv_36_21_weigths_cpd_rank_20_rot_n1.txt'};
    select_idxs{2,1} =[];
    fb{2,2} = {'separablefilters/mexico/fb_Mexico_uv_36_11_sep_cpd_rank_20_rot_n1.txt';...
               'separablefilters/mexico/fb_Mexico_uv_36_11_weigths_cpd_rank_20_rot_n1.txt'};
    select_idxs{2,2} =  []; 
    % channel 3 filters (v)
    %leanred
    fb{3,1} =  {'separablefilters/mexico/fb_Mexico_uv_36_21_sep_cpd_rank_20_rot_n1.txt';...
                'separablefilters/mexico/fb_Mexico_uv_36_21_weigths_cpd_rank_20_rot_n1.txt'};
    select_idxs{3,1} =[];
    fb{3,2} = {'separablefilters/mexico/fb_Mexico_uv_36_11_sep_cpd_rank_20_rot_n1.txt';...
               'separablefilters/mexico/fb_Mexico_uv_36_11_weigths_cpd_rank_20_rot_n1.txt'};
    select_idxs{3,2} =  [];
    n_fb_per_channel = [2 2 2];
    histeq_color = 0;
    smooth_init = 0;
    pooling_type = [];
    pooled_features_per_channel_idxs{1} = [];
    pooled_features_per_channel_idxs{end+1} = [];
    scale_factor_filters = [1 1 1 1 1];

    %% Create Multi-channel Image

    if ( max(max(max(input_img))) < 1.0 )
        error('This Function Expects Image Range [0~255]');
    end

    input_img = double(input_img);

    listDataFormat = strsplit(param.DataFormat, '_');
    output_img = zeros(size(input_img,1), size(input_img,2), 0);
    for idxDataFormat = 1:length(listDataFormat)
        switch lower(listDataFormat{idxDataFormat})
          case 'gray'
            %% Grayscale
            %             output_img = rgb2gray(input_img);
% $$$             cur_out_img = mean(input_img,3); %stupid greyscale conversion, needs to be fixed
            cur_out_img = 255.0*rgb2gray(input_img/255.0); % Note that division and multiplication is
                                                           % performed here due to the way
                                                           % matlab rgb2gray works
            output_img = cat(3,output_img,cur_out_img);
          case 'rgb'
            %% RGB color image
            output_img = cat(3,output_img,input_img);
          case 'grad'
            %% Image Gradients
            gx = zeros(size(input_img));
            gy = zeros(size(input_img));
            for ii=1:3
                [gx(:,:,ii), gy(:,:,ii)] = derivative5(double(input_img(:,:,ii)), 'x', 'y');  
            end
            mag = gx.*gx + gy.*gy;
            mag = sqrt(mag);
            [mag, magMaxIdx] = max(mag,[],3);
            % make indice mat to copy only max mag stuff
            indMat = (magMaxIdx-1) * size(mag,1)*size(mag,2) + reshape(1:size(mag,1)*size(mag,2),size(mag(:,:,1)));
            gx = gx(indMat)*0.5 + 128.0;
            gy = gy(indMat)*0.5 + 128.0;
            gradImg = cat(3,gx,gy);
            gradImg = cat(3,gradImg,mag);
            output_img = cat(3,output_img,gradImg);
          case 'desc'
            %% Descriptor Fields by Alberto
% $$$             gray_img = mean(input_img,3); % May need fixing
            gray_img = 255.0*rgb2gray(input_img/255.0); % Note that division and multiplication
                                                        % is performed here due to the way matlab
                                                        % rgb2gray works
            smooth_img = SmoothAndNormalize(gray_img,3.0);
            [xDerPos, xDerNeg, yDerPos, yDerNeg] = ComputeSignedDerivatives(smooth_img);
            output_img = cat(3,xDerPos,xDerNeg,yDerPos,yDerNeg);
            output_img = output_img * 20.0; % This multiplication is here
                                            % just to keep the values roughly
                                            % in the same range with others...
          case 'sym'
            %% Symmetry Features from Yannick
            [symS, symG] = image2SymFeat(mean(input_img,3));
            symG = symG*255.0;
            symS = symS + 128.0;
            symImg = cat(3, symS, symG);
            output_img = cat(3,output_img,symImg);
          case 'luv'
            %% LUV colorspace image
            % luv component
            J = rgbConvert(im2single(uint8(input_img)),'luv');
            %             luvImg = RGB2Luv(input_img/255.0);
            J=J*270.0; J(:,:,2)=J(:,:,2)-88.0; J(:,:,3)=J(:,:,3)-134.0;
            luvImg=double(J);
            luvImgReScaled = luvImg;
            luvImgReScaled(:,:,1) = luvImg(:,:,1)*255.0/100.0; % scaling for l
            luvImgReScaled(:,:,2) = (luvImg(:,:,2) + 134.0) * 255.0 / 354.0; % scaling for u
            luvImgReScaled(:,:,3) = (luvImg(:,:,3) + 140.0) * 255.0 / 256.0; % scaling for v
            output_img = cat(3,output_img,luvImgReScaled);
          case 'l'
            %% LUV colorspace image
            % luv component
            J = rgbConvert(im2single(uint8(input_img)),'luv');
            %             luvImg = RGB2Luv(input_img/255.0);
            J=J*270.0; J(:,:,2)=J(:,:,2)-88.0; J(:,:,3)=J(:,:,3)-134.0;
            luvImg=double(J);
            luvImgReScaled = luvImg;
            L_image = luvImg(:,:,1)*255.0/100.0; % scaling for l
% $$$             luvImgReScaled(:,:,2) = (luvImg(:,:,2) + 134.0) * 255.0 / 354.0; % scaling for u
% $$$             luvImgReScaled(:,:,3) = (luvImg(:,:,3) + 140.0) * 255.0 / 256.0; % scaling for v
            output_img = cat(3,output_img,L_image); % 
            
          case 'sep'
            %% Separable Filter results as input (experimental)
            % apply them to the image
            img_luv = rgbConvert(im2single(uint8(input_img)),'luv');
            [img_sep,~] = compute_features_boundary_img(img_luv,fb,n_fb_per_channel,histeq_color,smooth_init,pooling_type,select_idxs,pooled_features_per_channel_idxs,scale_factor_filters(1));
            sepImg = zeros(size(input_img,1), size(input_img,2), 0);
            for idxChannel = 1:length(img_sep)
                sepImg = cat(3,sepImg,reshape(img_sep{idxChannel}, [size(input_img,1), size(input_img,2), size(img_sep{idxChannel},2)]));
            end
            output_img = cat(3,output_img,sepImg);
          case 'ac'
            error('AC option has been moved to the wrapper function');
          otherwise
            error([listDataFormat{idxDataFormat} ' Not Implemented!']);
            % output_img = input_img;
        end        
    end

end

% im2disp = gx(:,:,1);
%  imshow((im2disp - min(min(im2disp)))/(max(max(im2disp)) - min(min(im2disp))));