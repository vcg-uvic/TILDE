function [keypts, score_res] = getKeypoints_TILDEP(img_info, p)
    
    suffix = '';
    if (isfield(p, 'optionalTildeSuffix'))
        suffix = ['_' p.optionalTildeSuffix];
    end
    
    trainset_name = p.trainset_name;
    testset_name = p.testset_name;
    fixed_scale = 10;%scale of the kp


        name_our_orig = ['../filters/BestFilters' suffix '/Original/' trainset_name 'Med.mat'];

        file_prefix = img_info.full_feature_prefix;
        file_suffix = ['_dump.mat'];
        filter_res_file_name = [file_prefix '_Train_' trainset_name '_Test_' testset_name file_suffix];        
        
        brun_filter = ~exist(filter_res_file_name,'file');
        if(brun_filter)
            [ binary_res, score_res ] = ApplyLearnedELLFilter(img_info.image_color, -inf,  name_our_orig, false );
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