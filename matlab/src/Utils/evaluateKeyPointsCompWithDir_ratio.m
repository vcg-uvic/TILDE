%% evaluateKeyPointsCompWithDir_ratio.m --- 
% 
% Filename: evaluateKeyPointsCompWithDir_ratio.m
% Description: 
% Author: Yannick Verdie, Kwang Moo Yi
% Maintainer: Yannick Verdie, Kwang Moo Yi
% Created: Tue Jun 16 17:13:51 2015 (+0200)
% Version: 
% Package-Requires: ()
% Last-Updated: Tue Jun 16 17:13:57 2015 (+0200)
%           By: Kwang
%     Update #: 1
% URL: 
% Doc URL: 
% Keywords: 
% Compatibility: 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Commentary: 
% 
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Change Log:
% 
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% Copyright (C), EPFL Computer Vision Lab.
% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%% Code:


function [eval_result] = evaluateKeyPointsCompWithDir_ratio( trainset_name, testset_name, num_key, parameters)
global sRoot;  

%for params tuning
    s = rng;%save state of rand
    if exist('stateRand.mat','file')
        s = load('stateRand.mat');
        s = s.s;
        rng(s);
    end


%% get configuration =====================================================

fprintf('Setting up configuration\n');
nameFolder = testset_name;
%[p] = setup_config_test(nameFolder); % load parameters for specific dataset
p.dataset_name  = nameFolder;
p.trainset_name = trainset_name;
p.testset_name = testset_name;
p.omp_num_threads = '16';
p.rootTest = [sRoot '/../data/' p.dataset_name '/test'];
p.test_img_list_filename = fullfile(p.rootTest,'test_imgs.txt');
p.optionalTildeSuffix = parameters.optionalTildeSuffix;

setenv('OMP_NUM_THREADS', p.omp_num_threads);

% end get config =========================================================


%%prepare data =========================================================
    [imgs_list,imgs_no] = get_list(p.rootTest,p.test_img_list_filename);
    
    if (imgs_no < 1)
        errorR('WHAT ! no image...');
    end
    
   bHasHomography = false;
   if exist([p.rootTest '/homography.txt'], 'file') == 2
      lines = importdata([p.rootTest '/homography.txt']);
      bHasHomography = true;
      
      if (length(lines) ~= imgs_no/2)
          error('Homography is provided but does not correspond to the right number of images');
      end
   end
    
    Hs = cell(imgs_no/2,1);
    list_img_info = cell(imgs_no,1);
    for i_img = 1:imgs_no
        imgs{i_img} = imread(imgs_list{i_img});
        [pathstr,name,ext] = fileparts(imgs_list{i_img});
        idx = strfind(pathstr,'/');pt = pathstr(1:idx(end));
        imgs_c{i_img} = imread([pathstr(1:idx(end)) 'image_color/' name ext]);
        list_img_info{i_img}.image_gray = imgs{i_img};
        list_img_info{i_img}.image_color = imgs_c{i_img};
        list_img_info{i_img}.image_name = imgs_list{i_img};
        
        if ~exist([pathstr(1:idx(end)) 'features/'], 'dir')
            mkdir([pathstr(1:idx(end)) 'features/']);
        end
        full_name = [pathstr(1:idx(end)) 'features/' name];
        list_img_info{i_img}.image_gray = imgs{i_img};
        list_img_info{i_img}.full_feature_prefix = full_name;
        
        if (i_img<=imgs_no/2)%load only half (pair)
             Hs{i_img} = eye(3);
               
            if (bHasHomography)%has custom homography provided
                Hs{i_img} = importdata([p.rootTest '/' lines{i_img}]);
            end
        end
    end

    
    
    %%%%%here is how to compute the nb of keypoints
    if (num_key < 1)
        radiusHardcoded  = 5;
        magic_number = 10/8; % should make random's repeatability to the desired stuff
        imgArea = size(imgs_c{1},1)*size(imgs_c{1},2);
        num_key = magic_number * num_key * (imgArea) / (pi * radiusHardcoded*radiusHardcoded);
        num_key = uint16(num_key);
        disp(['-------------->' trainset_name '_run_on_' testset_name 'Comparison_' num2str(num_key)]);
        
    end


    %get the name of the method we are going to test....
    %method starting with _ are ignored
    list_method = dir('KeypointDetectors');
    isub = [list_method(:).isdir]; %# returns logical vector
    list_method = {list_method(isub).name}';
    list_method(ismember(list_method,{'.','..'})) = [];
    idx = cellfun(@(x) x(1) == '_', list_method);
    list_method(idx) = [];
    nNumMethods = size(list_method,1);
    
    if (nNumMethods == 0)
        error('no method to test, error !')
    end

    % prepare all method storage
    features = cell(1,size(list_method,1));
    for i_method = 1:nNumMethods
        features{i_method} = cell(1,imgs_no);
    end
    
    % detect keypoints for all images
    for i_img = 1:imgs_no
        for i_method = 1:nNumMethods
            keyptMethodName = list_method{i_method};
            features{i_method}{i_img} = eval(['getKeypoints_' keyptMethodName '(list_img_info{i_img},p)']);
            if (~isempty(features{i_method}{i_img}))
                if (any(features{i_method}{i_img}(6,:)==0))
                   warning(['A scale for ' keyptMethodName 'is zero ( on image ' num2str(i_img) ' ), this will certainly crash later in the code....']); 
                end
            end
        end
    end

    
    %%-----------sort features
    pa.nbFeat = num_key;%return the nbFeat highest feature (use score for the sorting)
    pa.bDoAdaptiveNMS = false;

    idx_sorted_method = 1;
    for i_method = 1:nNumMethods
        if strcmpi(list_method{i_method}, 'SIFT')
            % SIFT sorted by score
            features_sorted{1,idx_sorted_method} = sortFeaturesSS(features{i_method},pa);;
        else
            % Method sorted by score
            features_sorted{1,idx_sorted_method} = {};
            if prod(cellfun(@(x) ~isempty(x),features{i_method})) % if there are no empty elements
                features_sorted{1,idx_sorted_method} = sortFeaturesSS_noAbs(features{i_method},pa);
            end          
        end
        sorted_legend_str{idx_sorted_method}=[list_method{i_method}];
        idx_sorted_method = idx_sorted_method + 1;  
        display(sprintf('Sorting Keypoints for %s',sorted_legend_str{idx_sorted_method-1}));
    end
    nNumSortedMethods = idx_sorted_method - 1;

    %% Evaluation
    idx_ready = 1;
    for idx_sorted_method = 1:nNumSortedMethods
        %display(sprintf('Evaluating %s',sorted_legend_str{idx_sorted_method}));
        if ~isempty(features_sorted{1,idx_sorted_method})
            [ AUC{idx_ready}] = evaluate(features_sorted{1,idx_sorted_method},pa, imgs,Hs);
            legend_str{idx_ready} = sorted_legend_str{idx_sorted_method};
            idx_ready = idx_ready + 1;
        end
    end
    
    if (idx_ready == 1)
        error('something wrong!');
    end
        
    eval_result.legend_str = legend_str;
	eval_result.AUC = AUC;

fprintf('Program terminated normally.\n');

% end everithing
   rng(s);
end

function [fileList di] = getAllFiles(dirName)

  dirData = dir(dirName);      %# Get the data for the current directory
  dirIndex = [dirData.isdir];  %# Find the index for directories
  fileList = {dirData(~dirIndex).name}';  %'# Get a list of the files
  if ~isempty(fileList)
    validIndex = ~ismember(fileList,{'.','..','.DS_Store'});  
    fileList =  fileList(validIndex);
    fileList = cellfun(@(x) fullfile(dirName,x),...  %# Prepend path to files
                       fileList,'UniformOutput',false);

  end
  subDirs = {dirData(dirIndex).name};  %# Get a list of the subdirectories
  validIndex = ~ismember(subDirs,{'.','..','.DS_Store'});  %# Find index of subdirectories
                                               %#   that are not '.' or '..'
  for iDir = find(validIndex)                  %# Loop over valid subdirectories
    nextDir = fullfile(dirName,subDirs{iDir});    %# Get the subdirectory path
    fileList = [fileList; getAllFiles(nextDir)];  %# Recursively call getAllFiles
  end

  di = dirData(3:end);
  di(~dirIndex(3:end)) = '';
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% evaluateKeyPointsCompWithDir_ratio.m ends here
