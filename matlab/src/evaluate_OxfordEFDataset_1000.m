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