% Classification using Support Vector Machine (SVM) 

clear all; clc; 

load fisheriris
X = meas(1:100,1:2);
Y = [-ones(50,1); ones(50,1)];

% Knowledge points: 
% Boxconstraint is the parameter that controls the tradoff between margin and misclassification term
% Boxconstraint is the Beta parameter in the SVM formualtion showed on Slides page 33 of Lecture 7 - Support Vector Machine

% The KernelScale set the parameter for the RBF kernel. It refers to Gamma
% on the radial basis function (RBF) on Slides Page 43 of Lecture 7 - Support Vector Machine

% BoxConstraint ? for parameter tunning, you can do grid search in the range [1e-4,1e4].
% KernelScale ? for parameter tunning, you can do grid search in the range [1e-4,1e4].

% Train Linear SVM Model
SVMModel1 = fitcsvm(X,Y,'KernelFunction','linear', 'BoxConstraint', 10);

% Train Linear SVM + Standarize Feature Matrix (mean=0 & std=1)
SVMModel2 = fitcsvm(X,Y,'KernelFunction','linear', 'BoxConstraint', 10, 'Standardize',true);

% Train Nonlinear SVM with RBF Kernel
SVMModel3 = fitcsvm(X,Y, 'KernelFunction','RBF','KernelScale', 1, 'BoxConstraint',10);

% set 'KernelScale','auto', the algorithm will search for an appropriate value automatically 
SVMModel4 = fitcsvm(X,Y, 'KernelFunction','RBF','KernelScale','auto', 'BoxConstraint',10);
 

% set 'OptimizeHyperparameters' to 'auto', optimize RBF-SVM parameters {'BoxConstraint','KernelScale'} automatically  
SVMModel5 = fitcsvm(X,Y,'OptimizeHyperparameters','auto'); 

% set 'OptimizeHyperparameters' to 'all', optimize all possible parameters
% automatically, including linear, RBF, and polynomial SVM
SVMModel6 = fitcsvm(X,Y,'OptimizeHyperparameters','all'); 
 

