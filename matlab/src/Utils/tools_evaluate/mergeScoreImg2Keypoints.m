function [keypts] = mergeScoreImg2Keypoints(keypts, scoreImg)

    idx = sub2ind(size(scoreImg),round(keypts(2,:)),round(keypts(1,:)));
    scores = scoreImg(idx);
    keypts(5,:) = scores;

end