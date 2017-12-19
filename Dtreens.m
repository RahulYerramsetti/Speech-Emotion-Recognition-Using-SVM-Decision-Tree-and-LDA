

%% Classify an Observation Using a Decision Tree Classifier.

%%
function Dtree()
%% 

[T] =xlsread('Atrain.xlsx','Sheet3');
[~,id] = xlsread('Atrain.xlsx','Sheet3','N2:N141');
[~,idd] = xlsread('Atrain.xlsx','Sheet5','A1:A4');

N = size(T,1);

[x1,y1,z1] = meshgrid(0:0.1:8,0:0.1:8,0:0.1:8);
x1 = x1(:);
y1 = y1(:);
z1 = z1(:);
%% Decision Tree
%
% Another classification algorithm is based on a decision tree. A decision
% tree is a set of simple rules.  Decision trees are also
% nonparametric because they do not require any assumptions about the
% distribution of the variables in each class.  
%
% The |fitctree| function creates a decision tree. 
[S] =xlsread('Atrain.xlsx','Sheet4');
[~,id1] = xlsread('Atrain.xlsx','Sheet4','N2:N21');

t = fitctree(...
    T(1:end,1:13), ...
    id, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 100, ...
    'Surrogate', 'off', ...
    'ClassNames', idd);

t2 = fitctree(...
    T(1:end,11:13), ...
    id, ...
    'SplitCriterion', 'gdi', ...
    'MaxNumSplits', 100, ...
    'Surrogate', 'off', ...
    'ClassNames', idd);

%%
% To view how the decision tree method divides the plane. 
% visualize the regions assigned to each emotion.

[grpname1,node1] = predict(t2,[x1 y1 z1]);
 gscatter(x1,y1,grpname1,'mrgb','sod')
 title('Regions of different Emotions');
 figure;

 

[grpname,node] = predict(t,T(1:140,1:13));
ConfusionMatrix = confusionmat(id,grpname)
[grpname2,node2] = predict(t,S(1:20,1:13));


 e1=0;
for K1 = 1 :1: 20

 if (strcmp(id1(K1),grpname2(K1)))
     e1=e1+1;
     
 end
end
fprintf('Total correct classified emotions out of 20 = %d\n', e1);
Testing_accuracy=e1/20




 x=S(1:end,10:10);
 y=S(1:end,11:11);
 hold on
 gscatter(T(1:140,10), T(1:140,11),id,'mrgb','sod');
 gscatter(x,y,grpname2,'mrgb','X')
 title('Training and Test Points Emotion classification');
 hold off 
 figure;

 
 
%%
% Another way to visualize the decision tree is to draw a diagram of the
% decision rule and class assignments.

view(t,'Mode','graph');
%%

% To determine the emotion assignment for an observation, start at the
% top node and apply the rule. If the point satisfies the rule you take
% the left path, and if not you take the right path. Ultimately you reach
% a terminal node that assigns the observation to one of the emotion.
%

%% 
% Compute the resubstitution error and the cross-validation error for
% decision tree.
dtResubErr = resubLoss(t)

%%
% First use |cvpartition| to generate 10 disjoint stratified subsets.
cp = cvpartition(id,'KFold',10)


cvt = crossval(t,'CVPartition',cp);
dtCVErr = kfoldLoss(cvt)
%%
% For the decision tree algorithm, the cross-validation error
% estimate is significantly larger than the resubstitution error. This
% shows that the generated tree overfits the training set. In other words,
% this is a tree that classifies the original training set well, but
% the structure of the tree is sensitive to this particular training set so
% that its performance on new data is likely to degrade. It is often
% possible to find a simpler tree that performs better than a more
% complex tree on new data.  
%
% Try pruning the tree. First compute the resubstitution error for various
% subsets of the original tree. Then compute the cross-validation error for
% these sub-trees. A graph shows that the resubstitution error is overly
% optimistic. It always decreases as the tree size grows, but beyond a certain
% point, increasing the tree size increases the cross-validation error rate.

resubcost = resubLoss(t,'Subtrees','all');
[cost,secost,ntermnodes,bestlevel] = cvloss(t,'Subtrees','all');
plot(ntermnodes,cost,'b-', ntermnodes,resubcost,'r--')
figure(gcf);
title('Graph of misclassification error');
xlabel('Number of terminal nodes');
ylabel('Cost (misclassification error)')
legend('Cross-validation','Resubstitution')

%%
% Which tree should you choose? A simple rule would be to choose the tree
% with the smallest cross-validation error.  While this may be
% satisfactory, you might prefer to use a simpler tree if it is roughly as
% good as a more complex tree. For this example, take the simplest tree
% that is within one standard error of the minimum.  That's the default
% rule used by the |cvloss| method of |ClassificationTree|.
%
% You can show this on the graph by computing a cutoff value that is equal to
% the minimum cost plus one standard error.  The "best" level computed by the
% |cvloss| method is the smallest tree under this cutoff. (Note that bestlevel=0
% corresponds to the unpruned tree, so you have to add 1 to use it as an index
% into the vector outputs from |cvloss|.)

[mincost,minloc] = min(cost);
cutoff = mincost + secost(minloc);
hold on
plot([0 20], [cutoff cutoff], 'k:')
title('Generated Decision Tree');
plot(ntermnodes(bestlevel+1), cost(bestlevel+1), 'mo')
title('Simple Tree with small cross-validation error');
legend('Cross-validation','Resubstitution','Min + 1 std. err.','Best choice')
hold off

%%
% Finally, you can look at the pruned tree and compute the estimated
% misclassification error for it. 

pt = prune(t,'Level',bestlevel);
view(pt,'Mode','graph')

%%
cost(bestlevel+1)

end