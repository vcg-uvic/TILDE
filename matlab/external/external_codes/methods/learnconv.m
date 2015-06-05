function [] = learnconv(in,out)

    h = [-127,  -53,   49,  123,  123,   53,  -49, -123;
          -92,  -39,   35,   88,   88,   39,  -35,  -85; 
          -35,  -18,   11,   32,   32,   14,  -11,  -28; 
           21,    7,  -11,  -25,  -21,   -7,   14,   28; 
           56,   21,  -25,  -56,  -56,  -21,   25,   60; 
           67,   28,  -25,  -64,  -64,  -28,   25,   64; 
           60,   28,  -21,  -53,  -56,  -25,   18,   49; 
           53,   25,  -14,  -42,  -46,  -21,   14,   39]'; % transposed since it was originally C

    % read image
    img = imread(in);
    img = rgb2gray(img);

    % apply filter
    [score_res] = imfilter(double(img), h, 'symmetric', 'conv');
    [score_res,binary_res] = ApplyNonMax2Score(score_res, [], true);
    
    % sort by score
    key_idx = find(binary_res); 
    [y,x] = ind2sub(size(binary_res),key_idx);
    s = score_res(key_idx);
    [~, idx_sort] = sort(s,1,'descend');
    y = y(idx_sort(1:end));
    x = x(idx_sort(1:end));
    s = s(idx_sort(1:end));

    % define scale?
    size_filter = 4.0;
    a = ones(size(y)) * 1.0/(size_filter^2);
    b = zeros(size(y)) * 1.0/(size_filter^2);
    c = ones(size(y)) * 1.0/(size_filter^2);

    % save to file
    fid = fopen(out,'w');
    fprintf(fid, '1\n');
    fprintf(fid, '%d\n',length(s));
    for idxKey = 1:length(s)
        fprintf(fid, '%f %f %f %f %f\n', x(idxKey),y(idxKey),a(idxKey),b(idxKey),c(idxKey));
    end
    fclose(fid);
    % save score to file
    fid = fopen([out '.score'],'w');
    for idxKey = 1:length(s)
        fprintf(fid, '%e\n', s(idxKey));
    end
    fclose(fid);

    
end