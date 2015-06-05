function [ binary,score ] = pts2binary( I,J, score )
%turns binary image with score image into x,y,s combo

    binary = zeros(size(score));
    binary(sub2ind(size(score),round(I),round(J))) = 1;

end

