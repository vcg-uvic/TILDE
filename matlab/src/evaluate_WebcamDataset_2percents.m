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

Allrepeatability = computeKP(parameters);