%% Classify an Observation Using a SVM Classifier.
%%
% Find a line separating the polish speech data on slected emotion features
% mfcc, energy,zcr,pitch

% Load the data and select features for classification
function SVMns()

[T] =xlsread('Atrain.xlsx','Sheet3');
[~,id] = xlsread('Atrain.xlsx','Sheet3','N2:N141');

gscatter(T(1:140,1), T(1:140,2),id,'mrgb','osd');
title('Initial Data set of Features');

xdata = T(:,1:13);
group = id(1:end);
% Use a linear support vector machine classifier
%Trainin Data using function fitcecoc(..)

svmStruct = fitcecoc(xdata,group);

[S] =xlsread('Atrain.xlsx','Sheet4');
[~,em] = xlsread('Atrain.xlsx','Sheet4','N2:N21');
Xnew = S(1:20,1:13);


[S1] =xlsread('Atrain.xlsx','Sheet3');
[~,em1] = xlsread('Atrain.xlsx','Sheet3','N2:N141');
Xnew3 = S1(1:end,1:13);

%Classify Data using function predict()

label = predict(svmStruct,Xnew); 
label3 = predict(svmStruct,Xnew3); 

 e=0;
for K = 1 :1: 20

 if (strcmp(em(K),label(K)))
     e=e+1;
     
 end
end
fprintf('Total correct classified emotions out of 20 = %d\n', e);
Testing_accuracy=e/20

%confusion matrix for testing data 
 ConfusionMatrix = confusionmat(id,label3)
 [x1,y1,z1] = meshgrid(0:0.1:8,0:0.1:8,0:0.1:8);
 x1 = x1(:);
 y1 = y1(:);
 z1 = z1(:);
 
 xdata1 = T(1:140,11:13);
 svmStruct1 = fitcecoc(xdata1,group);
 label1 = predict(svmStruct1,[x1 y1 z1]);
 %plotting different regions
 figure;
 gscatter(x1,y1,label1,'mrgb','sod');
 title('Regions of different Emotions');

%plot the classification
figure;
hold on;
gscatter(T(1:140,1), T(1:140,2),id,'mrgb','sod');
gscatter(Xnew(:,1),Xnew(:,2),label,'mrgb','X');
title('Training and Test Points Emotion classification');
hold off 
temp=0;
end