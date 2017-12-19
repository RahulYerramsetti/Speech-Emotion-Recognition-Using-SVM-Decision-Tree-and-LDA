clc; clear all; close all;


ANSWER=questdlg('Choose selection method:','Speech Emotion Recognition',...
    'SVM','Decision Tree','LDA','SVM');
UseSVM=strcmp(ANSWER,'SVM');
UseDecisionTree=strcmp(ANSWER,'Decision Tree');
UseLDA=strcmp(ANSWER,'LDA');
pause(0.01);

if UseSVM
    
    SVMns();
end   
if UseDecisionTree
    
    Dtreens();
end
if UseLDA
    
    LDAns();
    
end