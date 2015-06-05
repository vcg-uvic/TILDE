function [failed] = mser(in, out)
   failed = false;
   global sRoot; 
    binary_full_path = [sRoot '/external/external_codes/methods/mser.ln'];
    
    if (exist(binary_full_path) ~= 2)
        failed = true;
        return;
    end
    
    
    
    in_full_path = in;%[cd '/' in];
    if ~strcmpi(in_full_path(end-2:end), 'png')
        in_full_path(end-2:end) = 'png';
    end
    out_full_path = out;%[cd '/' out];
    
    com = sprintf(['%s -t 2 -i %s -o %s'], binary_full_path, ...
                  in_full_path, out_full_path);
    
    display('running the binary for mser...');
% $$$     display(com);
% $$$     pause;
    system(com);
    display('ran the binary without any issue');

    % compute score
    img = imread(in_full_path);
    if size(img,3) == 3
        img = rgb2gray(img);
    end
    
    % compute the scalespace
    [~, ~, INFO1] = vl_covdet(single(img));
    % compute scores
    [f1, ~, ~] = loadFeatures(out);
    [dog_score1, feat_scale1] = getDOGScore(INFO1.css, f1');
    dog_score1 = abs(dog_score1);


    % save the scores
    score_file = fopen([out '.score'], 'w');
    for ii = 1:length(dog_score1)
        fprintf(score_file, '%f\n', dog_score1(ii));
    end
    fclose(score_file);

end