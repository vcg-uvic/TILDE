%% link.m --- 
% 
% Filename: link.m
% Description: Script to create soft links 
% Author: Kwang Moo Yi, Yannick Verdie
% Maintainer: Kwang Moo Yi
% Created: Fri Jun  5 15:22:44 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:11:49 2015 (+0200)
%           By: Kwang
%     Update #: 6
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

if isunix
    if ismac
        suffix = '_mac';
    else
        suffix = '_linux';
    end
    
    % Link OpenCV keypoints
    if exist([' ../external/external_codes/methods/' ...
               'opencvKeypointDetector', suffix], 'file')
        com = ['ln -s opencvKeypointDetector', suffix, ' ../external/' ...
               'external_codes/methods/opencvKeypointDetector'];
        system(com);
    end
    
    % Link sifer
    if exist([' ../external/external_codes/methods/sifer', suffix], 'file')
        com = ['ln -s sifer', suffix, ' ../external/external_codes/methods/sifer'];
        system(com);
    end
    
elseif ispc
   error('PC is not supported');     
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% link.m ends here
