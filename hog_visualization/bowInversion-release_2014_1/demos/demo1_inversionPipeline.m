function out = demo1_inversionPipeline
% function out = demo1_inversionPipeline
% 
% author: Alexander Freytag
% date  : 27-05-2014 ( dd-mm-yyyy )

    %% setup dataset containing a single image only.
    %tmp = dir('/Users/macx/Documents/MATLAB/deep-goggle-master/experiments/data/imagenet12-val/*.JPEG') ;
    %imnet = cellfun(@(x) fullfile('/Users/macx/Documents/MATLAB/deep-goggle-master/experiments/data/imagenet12-val/', x), {tmp(1:100).name}, 'uniform', false);
    %imwrite(im, fullfile(expPath, [expName '-recon.png']));
    
    %for i = 1:numel(images)
    %    images{i}
    %end
    
    
    tmp = dir('/Users/macx/Documents/MATLAB/deep-goggle-master/experiments/data/imagenet12-val/*.JPEG') ;
    
    for i = 406:500
        imnet_pic = fullfile('/Users/macx/Documents/MATLAB/deep-goggle-master/experiments/data/imagenet12-val',{tmp(i:i).name});
        dataset.images        = { imnet_pic };
        
    %dataset.labels           = [1];
    %dataset.labels_names     = {'Lena'};
    %dataset.labels_perm      = [1];
    %dataset.labels_org_names = {'Lena'};
    
    
    %% extract features
    
    indicesImages = 1;
    
    % setup non-specified default values for all variables
    settingsLocalFeat = setupVariables_LocalFeatureExtraction( [] );
    
    % overlapping blocks on dense grid?
    settingsLocalFeat.b_overlappingBlocks = true;
    
    % if overlapping - which stride?
    settingsLocalFeat.i_stepSizeX         = 5;
    settingsLocalFeat.i_stepSizeY         = 5;        
    
    % size of blocks in px
    settingsLocalFeat.i_blockSizeX        = 64;
    settingsLocalFeat.i_blockSizeY        = 64;
    
    % enable progressbar
    settingsLocalFeat.b_progressbar       = true;    
    
    % where to cache features on disk?
    dataCache = DataCache();
    dataCache.setCacheFile( fullfile(pwd, 'demos', 'cache.mat' ));
    dataCache.m_bAllowOverwrite = true;
    
    settingsLocalFeat.dataCache = dataCache;
    
    
    % call feature extraction
    myFeatures = extractFeatures( settingsLocalFeat, dataset, indicesImages );   
    
    %% compute codebook
    
    settingsClustering = setupVariables_Clustering ( [] );
    
    codebookMethod      = settingsClustering.codebookStrategies{ 1 }.mfunction;
    codebook.prototypes = codebookMethod(myFeatures, settingsClustering );
    
    %% invert computed prototypes
    myBlockSize = [settingsLocalFeat.i_blockSizeY, settingsLocalFeat.i_blockSizeX];
    
    codebook.invPrototypes = invertPrototypes ( codebook.prototypes, myBlockSize ) ;    
    
    %% compute BoW inversion
    settingsHoggleBow.settingsLocalFeat = settingsLocalFeat;

    out = hoggleBow ( codebook, dataset.images{1}, settingsHoggleBow );
    
    imnet_result = fullfile('/Users/macx/Documents/MATLAB/bowInversion-release_2014_1/data/imnet_result', {tmp(i:i).name});
    imnet_result = char((cell2mat(imnet_result)));
    
    imnet_pic = char((cell2mat(imnet_pic)));
    imnet_pic = imread(imnet_pic);
    pic_size = size(imnet_pic);
    pic_size = pic_size(1:2);
    
    imnet_irbow_pic = imresize(out.imgHoggleBow, pic_size);
    imwrite(imnet_irbow_pic, imnet_result);
    
    %% prepare output, if desired
    if ( nargout > 0 )
        out.dataset      = dataset;
        out.myFeatures   = myFeatures;
        out.codebook     = codebook;
    end
    end
end
