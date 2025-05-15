clear all;
close all;
clc;

% Dataset loading 
fprintf('\n 1.Dataset loading')
[file path]=uigetfile('dataset.xlsx');
[num,txt,inputFile]=xlsread([path file]);
figure('position',[20 50 1200 600]),uitable('Data',inputFile,'position',[20 45 1100 550]);
% Data Selection and Target Selection 
fprintf('\n 2.Data pre-processing')

data=num(:,1:8);
labelData=num(:,9);
FeaturesFile=data;
LabelFile=round(labelData);
figure
c1 = categorical(LabelFile,[0 1], {'Abnormal', 'Normal'});
h1 = histogram(c1, 'FaceColor', 'r')
title("Electric Vehicles Using Reinforcement Learning ")

fprintf('\n 3.Feature Selection')

SearchAgents_no=size(FeaturesFile,2);
Max_iteration=500;
lb=-100;
ub=100;
dim=10;
[Best_score,Best_pos,cg_curve]=Greedy(SearchAgents_no,Max_iteration,lb,ub,dim,FeaturesFile(3:end,:)');

figure,
semilogy(cg_curve,'Color','r')
title('Convergence curve')
xlabel('Iteration');
ylabel('Best score obtained so far');

axis tight
grid off
box on
legend('Greedy')

display(['The best solution obtained by ALO is : ', num2str(Best_pos)]);
display(['The best optimal value of the objective funciton found by ALO is : ', num2str(Best_score)]);

g_best=GAAlgorithm(FeaturesFile);


M = 20;
N = size(FeaturesFile,2);
MaxGen = 100;
Pc = Best_score;
Pm = g_best;
Er = 0.05;
visualization = 1;
[FitnessChromosome,GeneChromosome]  = GeneticAlgorithm (M , N, MaxGen , Pc, Pm , Er , visualization,FeaturesFile);
optimizedFile=FeaturesFile;

for ii=1:size(GeneChromosome,2)
    if GeneChromosome(1,ii)==1
        optimizedFile=[optimizedFile abs(optimizedFile(:,ii).*FitnessChromosome)];
    end
end
figure('Name','Optimized Features','position',[20 50 1200 600]),uitable('Data',optimizedFile,'position',[20 45 1100 550]);

disp('The best chromosome found: ')
GeneChromosome
disp('The best fitness value: ')
FitnessChromosome
%----------------------------------------------------------------------------------------------------------
data=num(:,1:8);
labelData=num(:,9);
[XTrain] = data;
labelData = categorical(LabelFile);
[YTrain]=labelData(1:200);

for i=1:200
    xt{i}=XTrain(i,:);
end
inputSize = 1;
numHiddenUnits = 100;
numClasses = 2;

layers = [ ...
    sequenceInputLayer(inputSize)
    bilstmLayer(numHiddenUnits,'OutputMode','last')
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];

maxEpochs = 500;
miniBatchSize = 32;

options = trainingOptions('adam', ...
    'ExecutionEnvironment','cpu', ...
    'GradientThreshold',1, ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest', ...
    'Shuffle','never', ...
    'Verbose',0, ...
    'Plots','training-progress');
[net,tr] = trainNetwork(xt,YTrain,layers,options);



miniBatchSize = 12;
YPred = classify(net,xt, ...
    'MiniBatchSize',miniBatchSize, ...
    'SequenceLength','longest');


%---------------------------------------------------------------------------------------------------------------
%data Splitting 
X=data;
Y=LabelFile;
%---------------------------------------------------------------------------------------------------------------
%XGBOOST Algorithm 
Mdl = XGBOOST(X,Y)
Mdl.predict(X)
YTrain_ = predict(Mdl, X);
res(1) = performance_metrices( YTrain_, Y );
% disp classifier statistics
kappa1 = num2str( getfield(res, {1}, 'kappa', {1}), '%.3f' );
accu1 = num2str( getfield(res, {1}, 'ACC', {1}), '%.3f' );
disp(' ');
disp( ['[TRAIN]: kappa ' kappa1 ', accu: ' accu1] );
disp(' ');
% plot confusion matrix
c = confusionmat(YTrain_, Y');
figure('ToolBar', 'none', ...
	'Units', 'pixels', ...
	'Position', [300 300 500 500]);
confusion_metrices(c, {'NON-ATTACK', 'ATTACK '});

performance = classperf(Y,YTrain_);
Accuracy = performance.Sensitivity *100;
Sensitivity = performance.Sensitivity *100;
Specificity = performance.Specificity *100;

rnames={'%'};
cnames = {'Accuracy','Sensitivity','Specificity'};
f=figure('Name','Performance Measures for XGBOOST ','NumberTitle','off');
t = uitable('Parent',f,'Data',[Accuracy,Sensitivity,Specificity],'RowName',rnames,'ColumnName',cnames);

figure('Name','Performance Measures for XGBOOST ','NumberTitle','Off','Color','White');
bar(1,Accuracy,0.5,'r') ; hold on ; bar(2,Sensitivity,0.5,'g') ; hold on ; bar(3,Specificity,0.5,'b') ;
set(gca, 'XTick',1:3, 'XTickLabel',{'Accuracy' 'Sensitivity' 'Specificity'},'fontsize',12,'fontname','Times New Roman','fontweight','bold');


YTest=YTrain;
YPred=double(YPred);
YTest=double(YTest);
i=2;
if YPred[i]==0
    msgbox('Normal  ');
elseif YPred[i]==1
    msgbox('Abnormal');     
end

e =YTest-YPred;
MSE = mse (e)
MSE
MAE = mae (e)
MAE
fprintf('MAE is %s\n', MAE); 
(YTest - YPred) ;
(YTest - YPred).^2;
mean((YTest - YPred).^2);
RMSE = sqrt(mean((YTest - YPred).^2));
RMSE
fprintf('RMSE is %s\n', RMSE); 
