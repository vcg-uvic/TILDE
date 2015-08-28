%% evaluate_WebcamDataset_2percents.m --- 
% 
% Filename: evaluate_WebcamDataset_2percents.m
% Description: 
% Author: Yannick Verdie, Kwang Moo Yi
% Maintainer: Yannick Verdie, Kwang Moo Yi
% Created: Tue Jun 16 17:11:36 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Fri Aug 28 14:34:36 2015 (+0200)
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

parameters.nameDataset = 'Webcam';%for saving at the end
parameters.models = {'Mexico', 'Panorama','Chamonix', 'StLouis', 'Courbevoie', 'Frankfurt'};
parameters.testsets = {'Mexico', 'Panorama', 'Chamonix', 'StLouis', 'Courbevoie', 'Frankfurt'};
parameters.optionalTildeSuffix = '2percents';
parameters.numberOfKeypoints  = {85,161,123,0.02,0.02,0.02};
parameters.repeatabilityType = 'RADIUS';

Allrepeatability = computeKP(parameters);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% evaluate_WebcamDataset_2percents.m ends here
