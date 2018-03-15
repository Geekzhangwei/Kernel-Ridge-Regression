%% Exercise 10
% In this exercise we will perform kernel ridge regression (KRR)
% on the data set. For this exercise, you will hold out 2=3 of data
% for training and report the test results on the remaining 1/3.

clear all;close all;clc;
%% step 1: add path and load data
addpath('Functions\'); % add path of the function files
load('boston.mat'); % load original data
data_X = boston(:,1:end - 1); % independent variable
data_y = boston(:,end); % dependent variable

%% step 2: set parameters
trialsNum = 20; % the random number 
gamma = 2.^(-40:-26);
sigma = 2.^(7:0.5:13);
[G, S] = meshgrid(gamma, sigma); % divide the grid
[mm, nn] = size(G);
G = G(:);S = S(:);
dimNum = length(S);

mse_train = zeros(trialsNum, 1); % preallocate storage
mse_test = zeros(trialsNum, 1); % preallocate storage
G_choice = zeros(trialsNum, 1); % preallocate storage
S_choice = zeros(trialsNum, 1); % preallocate storage
ids = zeros(trialsNum,1); % preallocate storage

%% step 3: train model and perform nonlinear regression 
disp('Start iterating, please wait......');
for ii = 1:trialsNum 
    fprintf('\n  Iteration %d/20 has done!\n',ii);
    %% (1) Perform KRR on the training set using K-fold cross-validation 
    [X_train, y_train, X_test,y_test] = splitData(data_X, data_y); % split train datas and test datas
    fold5Score = zeros(dimNum, 1);% preallocate storage
    for jj = 1:dimNum
        fold5Score(jj) = kFoldScore(X_train, y_train, G(jj), S(jj), 5); % use 5-fold cross-validation to compute
    end
    [val, id] = min(fold5Score);
    ids(ii) = id;
    gamma = G(id);
    sigma = S(id);
    
    %% (2) Compute MSE and log other data
    G_choice(ii) = gamma;
    S_choice(ii) = sigma;
    K_train = Kernel_mat2(X_train, X_train, sigma); % compute kernel of training datas
    alpha = kridgereg(K_train, y_train, gamma); % commpute alpha in eqn.(12)
    mse_train(ii) = dualcost(K_train, y_train, alpha);% compute MSE
    
    K_test = Kernel_mat2(X_test, X_train, sigma); % compute kernel of test datas
    mse_test(ii) = dualcost(K_test, y_test, alpha); % compute MSE
end
disp('End of iteration! Let us draw a picture and evaluate it.');

%% step 4: Plot the \cross-validation error" as a function of gamma and sigma.
gamma0 = 2.^(-40:-26);
sigma0 = 2.^(7:0.5:13);
plot_cross_validation_error(gamma0, sigma0, fold5Score, mm, nn);

%% step 5: Calculate the MSE on the training and test sets for the best gamma and sigma.
fprintf('\nThe mean and var of mse on the training datas are as follows:\n');
trainingResult = [mean(mse_train) var(mse_train)]
fprintf('\nThe mean and var of mse on the test datas are as follows:\n');
testResult = [mean(mse_test) var(mse_test)]









