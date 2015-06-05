function [feat nb dim] = loadFeatures(file)
    fid = fopen(file, 'r');
    dim = fscanf(fid, '%f',1);
    if dim == 1
        dim = 0;
    end
    nb = fscanf(fid, '%d',1);
    feat = fscanf(fid, '%f', [5+dim, inf]);
    % iv
    feat = feat';
    % end iv
    fclose(fid);
end




