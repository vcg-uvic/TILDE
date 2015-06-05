function [ I,J,S ] = binary2pts( binary,score )
%turns binary image with score image into x,y,s combo

    [I,J] = find(binary);
    S = score(sub2ind(size(binary),I,J));

end

