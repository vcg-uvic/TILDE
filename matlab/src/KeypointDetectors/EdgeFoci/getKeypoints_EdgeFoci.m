function [keypts] = getKeypoints_EdgeFoci(img_info, p)

    edgefoci_name = [img_info.full_feature_prefix '_EdgeFoci_keypoints.mat'];
    if ~exist(edgefoci_name, 'file')
        if ~ispc
             warning('EdgeFoci is TURNED OFF on LINUX and MAC (binary not provided by the authors)');
             keypts = [];
        else
            in = img_info.image_name;
            in = strrep(in, 'image_gray', 'image_color');
            out = [img_info.full_feature_prefix '_EdgeFoci_keypointstxt'];
            failed = runKeypointsEdgeFoci(in, out);
            
            if (failed)
                warning('EdgeFoci TURNED OFF (binary not found)');
                keypts = [];
                return;
            end

            [feat, ~, ~] = loadFeatures(out);
            feat = feat';
            score = load([out '.score']);

           % Get the scale 
            a = feat(3,:);
            b = feat(4,:);
            c = feat(5,:);
            % obtain scales
            scale = sqrt(a.*c - b.^2); % sqrt of determinant (sqrt of product of eigs)
            scale = 1./sqrt(scale); % inverse becuz it's actually inv of [a b; b c]

            keypts = [feat(1:2,:); zeros(5,size(feat,2))];
            keypts(5,:) = score';
            keypts(6,:) = scale';

            save(edgefoci_name, 'keypts', '-v7.3');
        end
    else
        loadkey = load(edgefoci_name);
        keypts = loadkey.keypts;
    end
end
