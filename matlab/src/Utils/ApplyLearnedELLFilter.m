function [ binary_res, score_res_final ] = ApplyLearnedELLFilter( input_color_image, threshold, nameELLFilter, bDisp )
%% Load Filter
param = 0;
load(nameELLFilter);
param = res.param;
param = rmfield(param, 'fMultiScaleList');

if(size(input_color_image,3) == 1)
    input_color_image = repmat(input_color_image, [1 1 3]);
end
input_preproc_image = PreProcessTrainImage(input_color_image, param);

[ binary_res, score_res_final ] = ApplyLearnedELLFilterWithPreProcImage_NoLoadFile( input_preproc_image, threshold, res, bDisp );

