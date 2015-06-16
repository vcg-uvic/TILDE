%% evaluate_OxfordEFDataset_1000.m --- 
% 
% Filename: evaluate_OxfordEFDataset_1000.m
% Description: 
% Author: Yannick Verdie, Kwang Moo Yi
% Maintainer: Yannick Verdie, Kwang Moo Yi
% Created: Tue Jun 16 17:10:45 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:11:09 2015 (+0200)
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


warning off backtrace

addpath('Utils');
global sRoot;
tmp = mfilename('fullpath');tmp =  strsplit(tmp, '/');tmp= tmp(1:end-2);
sRoot = strjoin(tmp,'/');
setup_path

parameters.nameDataset = 'OxfordEF';%for saving at the end
parameters.models = {'Chamonix'};
parameters.optionalTildeSuffix = 'Standard';
parameters.testsets = {'bark','bikes','boat','graf','leuven','trees','ubc', 'wall', 'notredame', 'obama', 'yosemite', 'paintedladies', 'rushmore'}; 
parameters.numberOfKeypoints  = {1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000,1000};

computeKP(parameters);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% evaluate_OxfordEFDataset_1000.m ends here
