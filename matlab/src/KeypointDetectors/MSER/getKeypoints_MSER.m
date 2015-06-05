function [keypts] = getKeypoints_MSER(img_info, p)

    mser_name = [img_info.full_feature_prefix '_MSER_keypoints.mat'];
    if ~exist(mser_name, 'file')
         if ismac
                warning('MSER is TURNED OFF on MAC (binary not provided by the authors)');
                keypts = [];
         else
                in = img_info.image_name;
                out = [img_info.full_feature_prefix '_MSER_keypointstxt'];
                failed = mser(in,out);
                
                if (failed)
                    warning('MSER is TURNED OFF (binary not found)');
                    keypts = [];
                    return;
                end

                bFeatureReady = false;
                numTry = 0;
                while ~bFeatureReady
                    try
                        [feat, ~, ~] = loadFeatures(out);
                        feat = feat';
                        score = load([out '.score']);
                        bFeatureReady = true;
                    catch
                        warning('retrying...');
                        pause(1);
                        numTry = numTry + 1;
                        if(numTry == 10)
                            error('max try!');
                        end
                    end
                end

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

                save(mser_name, 'keypts', '-v7.3');
         end
    else
        loadkey = load(mser_name);
        keypts = loadkey.keypts;
    end
end