function [keypts] = getKeypoints_LearnedConvolutional(img_info, p)

     fixed_scale = 4;%half of the filter size

    h = [-127,  -53,   49,  123,  123,   53,  -49, -123;
          -92,  -39,   35,   88,   88,   39,  -35,  -85; 
          -35,  -18,   11,   32,   32,   14,  -11,  -28; 
           21,    7,  -11,  -25,  -21,   -7,   14,   28; 
           56,   21,  -25,  -56,  -56,  -21,   25,   60; 
           67,   28,  -25,  -64,  -64,  -28,   25,   64; 
           60,   28,  -21,  -53,  -56,  -25,   18,   49; 
           53,   25,  -14,  -42,  -46,  -21,   14,   39]'; % transposed since it was originally C

    learnconv_name = [img_info.full_feature_prefix '_learnconv_keypoints.mat'];
    if ~exist(learnconv_name, 'file')
        % convolve to get response map
        [score_res] = imfilter(double(img_info.image_gray), h, 'symmetric', 'conv');
        [score_res,binary_res] = ApplyNonMax2Score(score_res, [], true);

        idx = find(binary_res);
        [I,J] = ind2sub(size(binary_res),idx);
        keypts = [J I zeros(size(I,1),3) repmat(fixed_scale,size(I,1),1)]';
        keypts = mergeScoreImg2Keypoints(keypts, score_res);

        % safety check to prevent race condition
        if ~exist(learnconv_name, 'file')
            save(learnconv_name, 'keypts', '-v7.3');
        end
    else
        % loop to prevent race condition
        bFileReady = false;
        while (~bFileReady)
            try
                loadkey = load(learnconv_name);
                keypts = loadkey.keypts;
                bFileReady = true;
            catch
                pause(rand*5+5); % try again in 5~10 seconds
            end
        end
    end

end
