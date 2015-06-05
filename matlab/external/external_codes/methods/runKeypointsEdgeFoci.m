function [failed] = runKeypointsEdgeFoci(in_file_name, out_file_name)
    failed = false;
  global sRoot;

    detector_path = [sRoot '/external/external_codes/methods/EdgeFociAndBiCE.exe'];


    if (exist(detector_path) ~= 2)
        failed = true;
        return;
    end
    
    in_file_full_path =  [in_file_name];
    out_file_full_path = [out_file_name '_raw.txt'];
    
    % if not exist, compute the keypoints
    if ~exist(out_file_full_path, 'file')
        
        com = sprintf('%s -mi -i %s -o %s', detector_path, in_file_full_path, out_file_full_path);
        system(com);
    end
    [feat, score, nb] = loadFeaturesEdgeFoci(out_file_full_path);

    % save keypoints to our desired structure
    saveEdgeFociKeys(feat, nb, [out_file_name]);
    saveEdgeFociScores(score, [out_file_name '.score']);

end

function [feat, score, nb] = loadFeaturesEdgeFoci(file)
    fid = fopen(file, 'r');
    dim = 2;
    nb = fscanf(fid, '%d',1);
    feat_raw = fscanf(fid, '%f', [5+dim, inf]);
    % iv
    feat = feat_raw(1:5,:)';
    score = feat_raw(6,:)';
    % end iv
    fclose(fid);
end

function [] = saveEdgeFociKeys(feat,nb,file)
    fid = fopen(file, 'w');
    fprintf(fid,'%d\n',1);
    fprintf(fid,'%d\n',nb);
    for idxKey = 1:size(feat,1)
        fprintf(fid, '%f %f %f %f %f\n', feat(idxKey,1),feat(idxKey,2),1.0/feat(idxKey,3),feat(idxKey,4),1.0/feat(idxKey,5));
    end
    fclose(fid);    
end

function [] = saveEdgeFociScores(score,file)
    fid = fopen(file, 'w');
    for idxKey = 1:size(score,1)
        fprintf(fid, '%f\n', score(idxKey));
    end
    fclose(fid);    
end
