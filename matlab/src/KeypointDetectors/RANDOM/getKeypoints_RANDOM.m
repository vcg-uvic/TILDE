function [keypts] = getKeypoints_RANDOM(img_info, p)

    fixed_scale = 1;%half of the filter size

    % load image
    curImg = img_info.image_color;

    % binary keypoint image
    binary_res = ones(size(curImg,1),size(curImg,2));
    binary_res(randperm(numel(binary_res),3000)) = 1; % max 3000 for efficiency in sorting later
    score_res = rand(size(binary_res));

    % turn to keypts
    idx = find(binary_res);
    [I,J] = ind2sub(size(binary_res),idx);
    keypts = [J I zeros(size(I,1),3) repmat(fixed_scale,size(I,1),1)]';
    keypts = mergeScoreImg2Keypoints(keypts, score_res);

end