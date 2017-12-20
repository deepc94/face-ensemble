list = {'dist_cos_normal','dist_cos_avg','dist_cos_max', ...
    'dist_L2_normal','dist_L2_avg','dist_L2_max'};
root = 'Results4';
% 842,141
% 333,18
indices = randperm(333,18);
figure;
for i=1:6
    name = fullfile(root,strcat(list{i},'.mat'));
    load(name);
    matched = mat(indices,1);
    mismatched = mat(334:end,1);
    
    edges = linspace(min(matched), max(mismatched), 20);
    clf; hold on;
    histogram(matched, edges); histogram(mismatched, edges);
    legend;
    if list{i}(6) == 'c'
        xlabel('Cosine Distance');
    else
        xlabel('Euclidean Distance');
    end
    savefile = fullfile(root, strcat(list{i},'.png'));
    saveas(gcf,savefile);
end
close all;