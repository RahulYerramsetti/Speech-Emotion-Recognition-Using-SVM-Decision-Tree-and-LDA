%% Classify an Observation Using a Linear Discriminant Analysis Classifier.
%%

function LDAns()

[T] =xlsread('Atrain.xlsx','Sheet3');
[~,id] = xlsread('Atrain.xlsx','Sheet3','N2:N141');

N = size(T,1);

%% Linear Discriminant Analysis
% The |fitcdiscr| function can perform classification using different types
% of discriminant analysis. First classify the data using the default
% linear discriminant analysis (LDA).
%Fit discriminant analysis classifier
lda = fitcdiscr(T(1:end,1:13),id);
lda2 = fitcdiscr(T(1:end,11:13),id);
%Resubstitution prediction from a trained Gaussian process regression model
ldaClass = resubPredict(lda);
%%
% The observations with known class labels are usually called the training data.
% Now compute the resubstitution error, which is the misclassification error
% (the proportion of misclassified observations) on the training set.
ldaResubErr = resubLoss(lda);
%%
% You can also compute the confusion matrix on the training set. A
% confusion matrix contains information about known class labels and
% predicted class labels. Generally speaking, the (i,j) element in the
% confusion matrix is the number of samples whose known class label is
% class i and whose predicted class is j.  The diagonal elements represent
% correctly classified observations. 
[ldaResubCM,grpOrder] = confusionmat(id,ldaClass);
confusionmat(id,ldaClass)
%%
% misclassified by the linear discriminant function. You can see which ones
% they are by drawing X through the misclassified points.
bad = ~strcmp(ldaClass,id);
hold on;
plot(T(bad,1), T(bad,2), 'kx');
title('Misclassified points');
figure;
hold off;

%%
% The function has separated the plane into regions divided by lines, and
% assigned different regions to different species.  One way to visualize
% these regions is to create a grid of (x,y) values and apply the
% classification function to that grid.

[x1,y1,z1] = meshgrid(0:0.1:8,0:0.1:8,0:0.1:8);
 x1 = x1(:);
 y1 = y1(:);
 z1 = z1(:);

[S] =xlsread('Atrain.xlsx','Sheet4');
[~,idd] = xlsread('Atrain.xlsx','Sheet4','N2:N21');


j= predict(lda,S(1:20,1:13));
j1=predict(lda2,[x1 y1 z1]);

e=0;
for K = 1 :1: 20

 if (strcmp(idd(K),j(K)))
     e=e+1;
     
 end
end
fprintf('Total correct classified emotions out of 20 = %d\n', e);
Testing_accuracy=e/20


 
 x=S(1:20,10:10);
 y=S(1:20,11:11);

hold on
gscatter(T(1:140,10), T(1:140,11),id,'mrgb','sod');
gscatter(x,y,j,'mrgb','X')
title('Training and Test Points Emotion classification');
hold off
figure;
gscatter(x1,y1,j1,'mrgb','sod')
title('Regions of different Emotions');

%% 
% You have computed the resubstitution error. Usually people are more
% interested in the test error (also referred to as generalization error),
% which is the expected prediction error on an independent set. In fact,
% the resubstitution error will likely under-estimate the test error.
%
% In this case you don't have another labeled data set, but you can
% simulate one by doing cross-validation. A stratified 10-fold
% cross-validation is a popular choice for estimating the test error on
% classification algorithms. It randomly divides the training set into 10
% disjoint subsets. Each subset has roughly equal size and roughly the same
% class proportions as in the training set. Remove one subset, train the
% classification model using the other nine subsets, and use the trained
% model to classify the removed subset. You could repeat this by removing
% each of the ten subsets one at a time.
%
% Because cross-validation randomly divides data, its outcome depends on
% the initial random seed. To reproduce the exact results in this example,
% execute the following command:

rng(0,'twister');

%%
% First use |cvpartition| to generate 15 disjoint stratified subsets.
cp = cvpartition(id,'KFold',15)
%%
% The |crossval| and |kfoldLoss| methods can estimate the misclassification
% error for LDA using the given data partition |cp|.
%
% Estimate the true test error for LDA using 15-fold stratified
% cross-validation.

cvlda = crossval(lda,'CVPartition',cp);
ldaCVErr = kfoldLoss(cvlda)
%%
% The LDA cross-validation error has the same value as the LDA
% resubstitution error on this data.


end