function [dog_score, feat_scale] = getDOGScore(css, feat)

    if size(feat,1) == 5
        % parse x,y,a,b,c for easier coding
        x = feat(1,:)'-1;
        y = feat(2,:)'-1;
        a = feat(3,:)';
        b = feat(4,:)';
        c = feat(5,:)';

        % obtain scales
        scale = sqrt(a.*c - b.^2); % sqrt of determinant (sqrt of product of eigs)
        scale = 1./scale; % inverse becuz it's actually inv of [a b; b c]
    elseif size(feat,1) == 3
        x = feat(1,:)'-1;
        y = feat(2,:)'-1;
        scale = feat(3,:)';
    end
    feat_scale = scale;

	% do the log to get scale
	scale = scale / css.sigma0;
	log_scale = log2(scale);

	% round up to get o
	o = round(log_scale);
	% check o range
	o = min(max(css.firstOctave, o), css.lastOctave);

	% round up the remaining to get s
	s = round((log_scale-o)*css.octaveResolution);
	% check s range
	s = min(max(css.octaveFirstSubdivision, s), css.octaveLastSubdivision);

    % find overlapping subdivisions
    subdivision_overlap = css.octaveLastSubdivision - css.octaveFirstSubdivision + 1 - css.octaveResolution;
    idx_overlap = find(o ~= css.firstOctave & s < css.octaveFirstSubdivision + subdivision_overlap);
    
%     bUseHigherResolution = true;
%     if bUseHigherResolution
        % use higher resolution
        o(idx_overlap) = o(idx_overlap) - 1;
        s(idx_overlap) = s(idx_overlap) + css.octaveResolution;        

        % get the indices
        o_idx = o - css.firstOctave + 1;
        s_idx = s - css.octaveFirstSubdivision + 1;

        % get new coordinates for the corresponding octave
        x_o = round(x .* 2.^(-o))+1;
        y_o = round(y .* 2.^(-o))+1;
        
        % get the dog score
        dog_score = zeros(size(x_o,1),1);
        for idxFeat = 1:size(x_o,1)
            % boundary check
            x_o(idxFeat) = min(max(x_o(idxFeat), 1), size(css.data{o_idx(idxFeat)},2));
            y_o(idxFeat) = min(max(y_o(idxFeat), 1), size(css.data{o_idx(idxFeat)},1));
            try
                dog_score(idxFeat) = css.data{o_idx(idxFeat)}(y_o(idxFeat), x_o(idxFeat), s_idx(idxFeat));
            catch err
                display('something wrong');
            end
        end
%     else
%         % compute both
%         o_1 = o; o_2 = o;
%         s_1 = s; s_2 = s;
%         % subdivision in higher resolution
%         o_1(idx_overlap) = o_1(idx_overlap) - 1;
%         s_1(idx_overlap) = s_1(idx_overlap) + css.octaveResolution;        
%         % o_2 is left as subdivision in lower resolution
%         
%         %% for o_1
%         % get the indices
%         o_idx = o_1 - css.firstOctave + 1;
%         s_idx = s_1 - css.octaveFirstSubdivision + 1;
% 
%         % get new coordinates for the corresponding octave
%         x_o = round(x .* 2.^(-o_1))+1;
%         y_o = round(y .* 2.^(-o_1))+1;
% 
%         % get the dog score
%         dog_score_1 = zeros(size(x_o,1),1);
%         for idxFeat = 1:size(x_o,1)
%             dog_score_1(idxFeat) = css.data{o_idx(idxFeat)}(y_o(idxFeat), x_o(idxFeat), s_idx(idxFeat));
%         end
%         
%         %% for o_2
%         % get the indices
%         o_idx = o_2 - css.firstOctave + 1;
%         s_idx = s_2 - css.octaveFirstSubdivision + 1;
% 
%         % get new coordinates for the corresponding octave
%         x_o = round(x .* 2.^(-o_2))+1;
%         y_o = round(y .* 2.^(-o_2))+1;
% 
%         % get the dog score
%         dog_score_2 = zeros(size(x_o,1),1);
%         for idxFeat = 1:size(x_o,1)
%             dog_score_2(idxFeat) = css.data{o_idx(idxFeat)}(y_o(idxFeat), x_o(idxFeat), s_idx(idxFeat));
%         end        
%         %% combine
%         dog_score = sign(dog_score_1).* max(abs(dog_score_1), abs(dog_score_2));
% %         dog_score = dog_score_1;
%     end
end