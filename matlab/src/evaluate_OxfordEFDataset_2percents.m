%% evaluate_OxfordEFDataset_2percents.m --- 
% 
% Filename: evaluate_OxfordEFDataset_2percents.m
% Description: 
% Author: Yannick Verdie, Kwang Moo Yi
% Maintainer: Yannick Verdie, Kwang Moo Yi
% Created: Tue Jun 16 17:11:26 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:11:31 2015 (+0200)
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


warning off backtrace

addpath('Utils');
global sRoot;
tmp = mfilename('fullpath');tmp =  strsplit(tmp, '/');tmp= tmp(1:end-2);
sRoot = strjoin(tmp,'/');
setup_path

parameters.nameDataset = 'OxfordEF';%for saving at the end
parameters.models = {'Chamonix'};
parameters.optionalTildeSuffix = '2percents';
parameters.testsets = {'bark','bikes','boat','graf','leuven','trees','ubc', 'wall', 'notredame', 'obama', 'yosemite', 'paintedladies', 'rushmore'}; 
parameters.numberOfKeypoints  = {50, 174, 65, 101, 141, 175, 150, 175, 65, 39, 105, 65, 74};

computeKP(parameters);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% evaluate_OxfordEFDataset_2percents.m ends here
