function [keypts, score_res] = getKeypoints_TILDEP24(img_info, p)
global sRoot;

    suffix = '';
    if (isfield(p, 'optionalTildeSuffix'))
        suffix = ['_' p.optionalTildeSuffix];
    end

    trainset_name = p.trainset_name;
    testset_name = p.testset_name;
    fixed_scale = 10;%scale of the kp

    numFilters = 24;

        name_our_approx = ['BestFilters' suffix '/Approx/' trainset_name 'Med' num2str(numFilters) '.mat'];
        name_our_orig = ['BestFilters' suffix '/Original/' trainset_name 'Med.mat'];


        file_prefix = img_info.full_feature_prefix;
        file_suffix = ['_dump_approx' num2str(numFilters) '.mat'];
        filter_res_file_name = [file_prefix '_Train_' trainset_name '_Test_' testset_name file_suffix];        
        
        brun_filter = ~exist(filter_res_file_name,'file');
        if(brun_filter)
            input_color_image = img_info.image_color;
            clear param;load([sRoot '/filters/' name_our_orig],'res');
            param = res.param;
            delta = res.delta; %backup old delta from previous res structure
            res = load([sRoot '/filters/' name_our_approx]);
            % param = res.param;
            res.param = param;
            res.delta = delta;
            % res.newFilters = newFilters
            if isfield(param, 'fMultiScaleList')
                param = rmfield(param, 'fMultiScaleList');
            end

            if(size(input_color_image,3) == 1)
                input_color_image = repmat(input_color_image, [1 1 3]);
            end
            input_preproc_image = PreProcessTrainImage(input_color_image, param);

            [score_res, ~] = fastELLFiltering_approx(input_preproc_image, -Inf, res);
            fs = param.nBinSize;
            [score_res, max_img] = ApplyNonMax2Score(score_res, param);
            binary_res = max_img .* (score_res > -Inf);
            % Mutiplied fs with param.fScaling to consider scaling (25/04/2014 KMYI)
            binary_res(1:fs,:) = 0;
            binary_res(end-fs+1:end,:) = 0;
            binary_res(:,1:fs) = 0;
            binary_res(:,end-fs+1:end) = 0;
            parsavefilter(filter_res_file_name, score_res, binary_res);
        else
            display(' -- loaded dumped filter response');
            loadres = load(filter_res_file_name);
            score_res = loadres.score_res;
            binary_res = loadres.binary_res;
        end

        idx = find(binary_res);
        if(sum(sum(~isreal(score_res))))
        error(['Score Result for Our Filter has imaginary parts']);
        end
        [I,J] = ind2sub(size(binary_res),idx);
        keypts = [J I zeros(size(I,1),3) repmat(fixed_scale,size(I,1),1)]';
        keypts = mergeScoreImg2Keypoints(keypts, score_res);

end

function [] = parsavefilter(fname, score_res, binary_res)
    save(fname, 'score_res', 'binary_res');
end