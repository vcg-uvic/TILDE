function [score_img, binary_img] = ApplyNonMax2Score(score_img, param, bDoSimpleNonMax)
    
    if ~exist('param','var')
    	warning('using default nBinSize 21 and Shape Amos')
    	param.nBinSize = 21;
    	param.DesiredPosSpatialOutputShape = 'Amos';
    end

    if ~exist('bDoSimpleNonMax', 'var')
        bDoSimpleNonMax = true;
    end

    % Threshold Score first
%     score_img = max(score_img,0);

    % Apply NCC on the score
%     desShape = GetDesiredShape(param.DesiredPosSpatialOutputShape, param);
%     ncc_img = max(normxcorr2e(desShape,score_img, 'same'),0);
%     ncc_img = normxcorr2e(desShape,score_img, 'same')*0.5+0.5;
    
    % Multiply NCC res with original score
%     score_img = score_img .* ncc_img;
%     score_img = score_img + ncc_img;
%     score_img = ncc_img;


%     fs = param.nBinSize;
% 
%     dem = getDEM(score_img, fs);
%     score_img = score_img - dem;
    
    binary_img = score_img > 100;
    bDoSimpleNonMax = true;
    basic_nonMax = bDoSimpleNonMax;
    if (basic_nonMax)
%         score_img = max(score_img,0);
        score_img = vl_imsmooth( score_img, 2.0);
        binary_img = zeros(size(score_img));
        binary_img(vl_localmax(double(score_img))) = 1;
%         binary_img = score_img==imdilate(score_img,strel('square',3));
    else
        thr_minScore = 0;
        nbOctave = 3;
        nbScalePerOctave = 4;
        scaleSpace = zeros(size(score_img,1),size(score_img,2),nbOctave*nbScalePerOctave);
        score_img = score_img;%max(score_img,0);
        scaleSpace(:,:,1) = score_img;%/max(max(score_img))
        
        binary = zeros(size(score_img,1),size(score_img,2),nbOctave*nbScalePerOctave);
        b = zeros(size(score_img,1),size(score_img,2),1);
        b(vl_localmax(scaleSpace(:,:,1),thr_minScore))=1;
        binary(:,:,1) = b;
        
        for i=1:size(scaleSpace,3)-1
%             sigma_scale = sqrt((1.6*2^((i-1)/nbScalePerOctave))^2-0.25);
            sigma_scale = sqrt((1.6*2^((i-1)/nbScalePerOctave))^2-0.25);
%             scaleSpace(:,:,i+1) =  vl_imsmooth( score_img, sigma_scale)*(sigma_scale*sigma_scale);
            scaleSpace(:,:,i+1) =  vl_imsmooth( score_img, sigma_scale)*(sigma_scale*sigma_scale);
%             b = zeros(size(score_img,1),size(score_img,2),1);
%             b(vl_localmax(scaleSpace(:,:,i+1),thr_minScore))=1;
%             binary(:,:,i+1) = b;
        end
        
%         binary_num = reshape(binary, [size(binary,1)*size(binary,2), size(binary,3)]);
%         binary_num = sum(binary_num,1)';
%         binary_num = abs(binary_num - 100);
%         [~,I] = min(binary_num);

%         binary_img = sum(binary,3);
%         numkey = zeros(5,1);
%         for th = 1:5
%             numkey(th) = sum(sum(binary_img > th));
%         end
%         [~,th] = min(abs(numkey - 100*2));
%         binary_img = binary_img > 2;
        
%         max_idx = vl_localmax(scaleSpace);
%         [I,J,K] = ind2sub(size(scaleSpace),max_idx);
%         [S] = scaleSpace(max_idx);        

        b = zeros(size(scaleSpace));
        b(vl_localmax(scaleSpace))=1;
        binary_img = max(b,[],3);
        score_img = max(scaleSpace,[],3);
%         score_img = mean(scaleSpace,3);
        
%         binary_img = zeros(size(score_img));
%         score_img = zeros(size(score_img));
%         %find maxima accross scale space
% 
%         img = zeros(size(score_img,1),size(score_img,2));
%         for i=1:size(scaleSpace,3)-1
%             img(vl_localmax(scaleSpace(:,:,i),thr_minScore)) = img(vl_localmax(scaleSpace(:,:,i),thr_minScore)) ;
%         end
        
    end
    
    
    
    % smooth and do the stuff here

    
end