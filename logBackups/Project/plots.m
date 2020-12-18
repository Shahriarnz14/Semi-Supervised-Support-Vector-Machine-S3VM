close all;

data = {load('self50.mat') ;load('self100.mat');load('self200.mat');...load('self250.mat');
        load('self300.mat');load('self400.mat');...
        load('self600.mat');load('self1000.mat')};
    
    
labeled_size = zeros(length(data),1);
legend_text = cell(length(data),1);
for i = 1:length(data)
    labeled_size(i) = data{i}.train_size(1);
    legend_text{i} = ['|L|_{init} = ',num2str(labeled_size(i)),' Samples'];
end

%% Score Epochs and Reliability Epochs
figure;
subplot(1,2,1);
for i=1:length(data)
    plot(1-data{i}.score,'linewidth', 0.75); hold on;
end
xlabel('Epoch Number');
ylabel('Test Error');
title('(a) Test Prediction Error as Self-Training Progresses')
grid on
clr = get(gca,'colororder');
xlim([1,inf]);
% Starting SVM Accuracy
% for i=1:length(data)
%     plot(data{i}.score(1)*ones(length(data{i}.score)),'-.','color',clr(i,:));
% end
legend(legend_text);

subplot(1,2,2)
for i=1:length(data)
    plot(data{i}.reliability,'-.','linewidth', 0.75); hold on;
end
xlabel('Epoch Number');
ylabel('Prediction Confidence');
title('(b) Prediction Confidence for Inclusion of Un-labeled Data as Self-Training Progresses')
grid on
xlim([1,inf]);
legend(legend_text);

%% Final Prediction Accuracy
figure;
%subplot(2,2,4)
init_score  = zeros(length(data),1);
final_score = zeros(length(data),1);
for i=1:length(data)
    init_score (i) = data{i}.score(1);
    final_score(i) = data{i}.score(end);
    %final_score(i) = data{i}.score(max(find(data{i}.reliability < 0.95,1)-1,1));
end

% labeled_size = [labeled_size;3000];
% init_score   = [init_score  ;0.9];
% final_score  = [final_score ;0.9];

plot(labeled_size,init_score ,'--','linewidth', 0.75); hold on;
plot(labeled_size,final_score,'linewidth', 0.75);

yaxisTmp = [labeled_size',fliplr(labeled_size')];
inBtw1 = [init_score'+1/sqrt(4*3000),fliplr(init_score'-1/sqrt(4*3000))];
inBtw2 = [final_score'+1/sqrt(4*3000),fliplr(final_score'-1/sqrt(4*3000))];
%plot(yaxis,beamPressMean+beamPressStd,'-' ,'color', clr(1,:));
%plot(yaxis,beamPressMean-beamPressStd,'-' ,'color', clr(1,:));
fill(yaxisTmp,inBtw1,clr(1,:),'LineStyle','none'); alpha(0.15);
fill(yaxisTmp,inBtw2,clr(2,:),'LineStyle','none'); alpha(0.15);

xlabel('Number of Labeled Digits');
ylabel('Test Accuracy');
title('Test Prediction Accuracy for Varying Initial Number of Labeled Digits)')
grid on
legend('SVM (NO Un-labeled Digits)','Self-Training SVM with Un-labeled Digits (|L|+|U|=3000)');
xlim([-inf,inf]);


%% Table

accuracy_pairs = load('heatmapData_new.mat');
accuracy_pairs = accuracy_pairs.AA;
figure;
x_vals = {'0','1','2','6','3','5','8','4','7','9'};
heatmap(x_vals,x_vals,100-accuracy_pairs);
xlabel('MNIST Digits')
ylabel('MNIST Digits')
%h = colorbar;
%ylabel(h, 'Test Accuracy (%)')
title({'Heat Map of Test Accuracy Percetange for QS3VM - Classification Pairs';...
        '|L|_i=20 , |U|_i=580 , |L|+|U|=600'})
    
%% Tom Mitchell
X = [30,40,60,130,200];
y_SVM = [0.25,0.148 ,0.1398, 0.10, 0.073];
y_S3VM  = [0.125,0.105 ,0.10  , 0.09, 0.075];

plot(X,1-y_SVM ,'--','linewidth', 0.75); hold on;
plot(X,1-y_S3VM,'linewidth', 0.75);

xlabel('Number of Labeled Digits');
ylabel('Test Accuracy');
title({'Test Prediction Accuracy for Varying Initial Number of Labeled Digits (Q-S3VM)';...
        'Digits 4 and 9'})
grid on
legend('SVM (NO Un-labeled Digits)','Self-Training SVM with Un-labeled Digits (|L|+|U|=600)');
xlim([-inf,inf]);


