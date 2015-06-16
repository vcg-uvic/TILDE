%% setup_path.m --- 
% 
% Filename: setup_path.m
% Description: 
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi, Yannick Verdie
% Created: Tue Jun 16 17:14:08 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:14:13 2015 (+0200)
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


global bSetupPathFin
global sRoot;

doSetupPath = false;
if ~exist('bSetupPathFin','var')
    doSetupPath = true;
else
    if(isempty(bSetupPathFin) || bSetupPathFin ~= true)
        doSetupPath = true;
    end
end
if doSetupPath
    bSetupPathFin = true;

addpath(sRoot);
addpath(genpath([sRoot '/src/KeypointDetectors']));
addpath(genpath([sRoot '/src/Utils']));
addpath([sRoot '/filters']);
addpath([sRoot '/external/vlfeat-0.9.18/toolbox/']);
addpath([sRoot '/external/dollarToolbox']);
addpath(genpath([sRoot '/external/external_codes/']));

vl_setup;

%mkdir(sRoot,'resultAUCs');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% setup_path.m ends here
