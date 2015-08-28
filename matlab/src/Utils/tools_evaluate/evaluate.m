%% evaluate.m --- 
% 
% Filename: evaluate.m
% Description: 
% Author: Yannick Verdie, Kwang Moo Yi
% Maintainer: Yannick Verdie, Kwang Moo Yi
% Created: Tue Jun 16 17:14:18 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Fri Aug 28 14:42:28 2015 (+0200)
%           By: Kwang
%     Update #: 3
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


function [AUC] = evaluate(features, pb, imgs,Hs)

if ~exist('Hs', 'var')
    Hs = cell(1,size(features,2)/2);
    for i=1:size(features,2)/2
        Hs{i} = eye(3);
    end
end


    if (mod(size(features,2),2) ~= 0)
        warning('to spit the set of image in 2 equal parts, we need a power of 2 images !')
    end


    % re-do the cutting
    if (exist('pb','var'))
        if isfield(pb,'nbFeat')
            parfor i=1:size(features,2)
                features{i} = features{i}(:,1:min(pb.nbFeat,size(features{i},2)));
            end
        end
    end

    %create n independant kdtree
     parfor i=1:size(features,2)
        kd{i} = KDTreeSearcher(features{i}(1:2,:)');
     end

%                 %do it n times
    score = cell(1,size(features,2)/2);
     for i=1:size(features,2)/2
         
         range = 5;                     % this is by default radius based matching. not used for
                                        % oxford
         repeatabilityType = pb.repeatabilityType;
         
         %for standart, we should not use growAndScore
         %pic i vs pic i+n/2
         [score{i}] = growAndScore(features{i},features{i+size(features,2)/2},imgs{i}, Hs{i}, range, repeatabilityType);
     end
    AUC = mean(cell2mat(score),2);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% evaluate.m ends here
