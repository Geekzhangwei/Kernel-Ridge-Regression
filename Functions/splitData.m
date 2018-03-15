%% This function randomly splits a given sample set and ground truth y,
% to 2:1 ratio for training and testing sets.

% input£º
  % X£ºindependent variable
  % y£ºdependent variable
  
% output:
  % X_train: independent variable in the training data
  % y_train: dependent variable in the training data
  % X_test: independent variable in the test data
  % y_test: dependent variable in the test data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
function [X_train, y_train, X_test,y_test] = splitData(X, y)
%% Shuffle data
ids = randperm(size(X,1));
X = X(ids,:);y = y(ids);
%% 2:1 ratio split hardcoded
train_frac = 2/3;
test_frac = 1/3;
%% divide data
X_train = X(1:floor(size(X,1)*train_frac),:);
y_train = y(1:floor(size(X,1)*train_frac));
X_test = X(floor(size(X,1)*test_frac) + 1:end,:);
y_test = y(floor(size(X,1)*test_frac) + 1:end);
end