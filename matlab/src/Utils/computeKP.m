function Allrepeatability = computeKP(parameters)

    nameDataset = parameters.nameDataset;%for saving at the end
    models = parameters.models;
    testsets = parameters.testsets;
    numberOfKeypoints  = parameters.numberOfKeypoints;
    
    combinations = cell(length(models)*length(testsets),2);
    idxComb = 1;
    for idxModel = 1:length(models)
        for idxTestSet = 1:length(testsets)
            combinations{idxComb,1} = models{idxModel};
            combinations{idxComb,2} = testsets{idxTestSet};
            combinations{idxComb,3} = numberOfKeypoints{idxTestSet};
            idxComb = idxComb + 1;
        end
    end

    Allrepeatability = cell(size(combinations,1),1);
    for idxComb = 1:size(combinations,1)

        res = evaluateKeyPointsCompWithDir_ratio(combinations{idxComb,1}, combinations{idxComb,2}, combinations{idxComb,3}, parameters);

        repeatability = cell2mat(res.AUC);
        Allrepeatability{idxComb} = repeatability;

        %=========================================for display
        colors = hsv(length(res.legend_str));
        figure;
        for i=1:length(res.legend_str)
            bar(i,repeatability(i), 'facecolor', colors(i,:));
            hold on;
        end
        hold off;
        title(['Trained with ' combinations{idxComb,1} ' and tested on ' combinations{idxComb,2}])
        set(gca, 'XTick', 1:length(res.legend_str))
        legend(res.legend_str, 'Location','SouthEast');
        text(1:length(res.legend_str),repeatability',num2str(repeatability','%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom')
        drawnow;
    end
    Allrepeatability = cell2mat(Allrepeatability);

    idxComb2 = 1;
    clear AllrepeatabilityCross
    for idxModel = 1:length(models)
        clear repeatabilityCross
        idxComb1 = 1;
        %repeatabilityCross = zeros(length(testsets)-1,size(Allrepeatability,2));
        for idxTestSet = 1:length(testsets)
            if (strcmp(models{idxModel},testsets{idxTestSet}) == 0)
                repeatabilityCross{idxComb1,1} = Allrepeatability(idxComb2,:);
                idxComb1 = idxComb1 + 1;
            end
            idxComb2 = idxComb2 + 1;
        end
        
        repeatabilityCross = mean(cell2mat(repeatabilityCross),1);
        AllrepeatabilityCross{idxModel,1} = repeatabilityCross;

        %=========================================for display
        colors = hsv(length(res.legend_str));%we expect the legend not to change....
        figure;
        for i=1:length(res.legend_str)
            bar(i,repeatabilityCross(i), 'facecolor', colors(i,:));
            hold on;
        end
        hold off;
        title(['Trained with ' models{idxModel} ' and tested ' num2str(idxComb1-1) ' others (average))'])
        set(gca, 'XTick', 1:length(res.legend_str))
        legend(res.legend_str, 'Location','SouthEast');
        text(1:length(res.legend_str),repeatabilityCross',num2str(repeatabilityCross','%0.2f'),'HorizontalAlignment','center','VerticalAlignment','bottom')
        drawnow;
        
    end
    AllrepeatabilityCross = cell2mat(AllrepeatabilityCross);
    
    save(['results_' nameDataset '_' num2str(numberOfKeypoints{1})]);
end