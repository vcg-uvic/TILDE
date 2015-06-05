function [score] = growAndScore(feat1,feat2,img, H, range, repeatabilityType)

if ~exist('H', 'var')
	 H = eye(3);
end

if ~exist('range', 'var')
	range = 5; 
end

if ~exist('repeatabilityType','var')
    repeatabilityType = 'RADIUS';
end
   
    im1 = img;
    im2 = img;
    
    
    if (any(feat1(6,:) == 0))
        error('A feature scale is zero')
    end

%range is for fast evaluation
%nrange is for standard 40%overlaps evaluation
    f1 = zeros(5,size(feat1,2));
    f1(3,:) = 1.0./(feat1(6,:).^2);
    f1(5,:) = 1.0./(feat1(6,:).^2);
    f1(1:2,:) = feat1(1:2,:);

    f2 = zeros(5,size(feat2,2));
    f2(3,:) = 1.0./(feat2(6,:).^2);
    f2(5,:) = 1.0./(feat2(6,:).^2);
    f2(1:2,:) = feat2(1:2,:);

    if (size(f1,2) == 0 || size(f2,2) == 0)
        repeat = zeros(1,6);
    else
        [~,repeat,~,~,~,~]=repeatability_noLoadFile(f1,f2,H,im1,im2, range, repeatabilityType);
    end
    score = repeat(4);%pick 4 for 40% overlaps

%     warning(['The current Evaluation does FAST-Like evaluation with ' ...
%              'fixed ratio of 5!']);
end