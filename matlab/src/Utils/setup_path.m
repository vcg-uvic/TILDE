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
