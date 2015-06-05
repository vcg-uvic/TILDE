%% Example call for the SFOP keypoint detector.
%
% If you want to match detected keypoints, you have to compute descriptors.
% You can compute SIFT descriptors using the program "extract_features"
% from Krystian Mikolajczyk's website
% 	http://www.robots.ox.ac.uk/~vgg/research/affine/detectors.html
% Proceed as follows, given an example image test.jpg:
% (1) Detect SFOP keypoints from matlab
% 	  sfop('test.jpg','test.sfop');
% (2) Convert the image to pgm -> test.pgm
% (3) Compute descriptors:
%     extract_features -p1 test.sfop -o1 test.desc -sift -i test.pgm
% Matches can then be found by searching nearest neighbours in the space of
% descriptors, which are contained in columns 6:end of file test.desc.
%
% For further information refer to
%   http://www.ipb.uni-bonn.de/uploads/tx_ikgpublication/foerstner09.detecting.pdf
% and cite as
%   @inproceedings{
%     foerstner*09:detecting,
%     author={W. F\"orstner and T. Dickscheid and F. Schindler},
%     title={Detecting Interpretable and Accurate Scale-Invariant Keypoints},
%     booktitle={12th IEEE International Conference on Computer Vision (ICCV'09)},
%     address={Kyoto, Japan},
%     year={2009}
%   }
%
% Licence:
%   For internal use only.
%
% Warranty:
%   No warranty for validity of this implementation.
%
% Authors:
%   Wolfgang Foerstner, Timo Dickscheid, Falko Schindler
%   Department of Photogrammetry
%   Institute of Geodesy and Geoinformation
%   University of Bonn
%   Bonn, Germany
%
% Contact person:
%   Falko Schindler (falko.schindler@uni-bonn.de)
%
% Copyright 2009-2011

%% detect sfop features with optimal alpha (general spiral-type features)
disp('Running keypoint detector for general spiral-type features');
sfop('../examples/siemens_sm.png', 'siemens_sm.alphamin.sfop');

%% detect only junctions (alpha=0) by setting type=0
disp('Running keypoint detector for junction-type features');
sfop('../examples/siemens_sm.png', 'siemens_sm.alpha0.sfop',  'type',  0);

%% detect only circular features (alpha=90) by setting type=90
disp('Running keypoint detector for circular symmetric features');
sfop('../examples/siemens_sm.png', 'siemens_sm.alpha90.sfop', 'type', 90);

%% show all three sets (in yellow, with line width 1)
figure('name', 'SFOP results - general spiral type features');
showFeatures('../examples/siemens_sm.png', 'siemens_sm.alphamin.sfop');
figure('name', 'SFOP results - junction type features');
showFeatures('../examples/siemens_sm.png', 'siemens_sm.alpha0.sfop');
figure('name', 'SFOP results - circular symmetric features');
showFeatures('../examples/siemens_sm.png', 'siemens_sm.alpha90.sfop');

%% clean up
% delete siemens_sm.alphamin.sfop
% delete siemens_sm.alpha0.sfop
% delete siemens_sm.alpha90.sfop
