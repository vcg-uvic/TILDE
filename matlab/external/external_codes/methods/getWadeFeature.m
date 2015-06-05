function [feat1] = getWadeFeature(img1, thrs, itr, customPath, minNbFeat)
if ~exist('thrs','var')
    thrs = 0.5;
end

if ~exist('customPath','var')
    customPath = '.';
end

if ~exist('minNbFeat','var')
    minNbFeat = 100;
end

if ~exist('itr','var')
    itr = 0;
end

fixed_size = 1;

resFileName = [img1 '.wade.txt'];
    full_path_exec = [customPath '/../external/external_codes/WaveDetector'];
    
    if (exist(full_path_exec) ~= 2)
        feat1 = [];
        return;
    end
    
command = [full_path_exec ' -r ' num2str(thrs) ' -i ' img1 ' --saveSharpness -o ' resFileName];
[status,cmdout] = system(command);

if status ~= 0
    error(['Could not run WADE ! | ' cmdout])
end

res1 = textread(resFileName, '',  'emptyvalue', NaN);
res1 = res1(3:end,1:end-2);

[m n] = size(res1);
disp(['Found #' num2str(m) ' Keypoints']);
if m < minNbFeat && itr < 10
    disp('not enough keypoint, lower threshold...');
	feat1 = getWadeFeature(img1, 0.8*thrs, itr+1, customPath, minNbFeat);
else
	%feat1 = [res1(:,1:3) zeros(m,1) res1(:,4) fixed_size*ones(m,1) zeros(m,1)]';
    feat1 = [res1(:,1:2) zeros(m,2) res1(:,4) sqrt(1./res1(:,3)) zeros(m,1)]';
end

if itr == 20
    error('not enough keypoints in WADE (and lower threshold does not help)');
end

%disp(feat1)

end
