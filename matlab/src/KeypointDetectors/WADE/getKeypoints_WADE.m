function [keypts] = getKeypoints_WADE(img_info, p)

    wade_name = [img_info.full_feature_prefix '_wade_keypoints.mat'];
    if ~exist(wade_name, 'file')
        if ismac
            warning('WADE is TURNED OFF on MAC (binary not provided by the authors)');
            keypts = [];
        else
            keypts = getWadeFeature(img_info.image_name);
            
            if isempty(keypts)
                warning('WADE is TURNED OFF (binary not found)');
                keypts = [];
                return;
            end
            
            wade_img = zeros(size(img_info.image_color,1),size(img_info.image_color,2));
            for idxFeat = 1:size(keypts,2)
                wade_img(round(keypts(2,idxFeat)),round(keypts(1,idxFeat))) = 255;
            end
            display(['Wade Features #' num2str(size(keypts,2))]);
            imwrite(wade_img, 'debug_wade_img.bmp');
            save(wade_name, 'keypts', '-v7.3');
        end
    else
        loadkey = load(wade_name);
        keypts = loadkey.keypts;
    end        
end