function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

choices = [ 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
C_choices = choices;
sigma_choices = choices;
count = 1;
min_error = ones(3, 1);
for C_try = C_choices
  for sigma_try = sigma_choices
    printf("Num %d trying, when C = %f, sigma = %f", count++, C_try, sigma_try);
    model= svmTrain(X, y, C_try, @(x1, x2) gaussianKernel(x1, x2, sigma_try)); 
    predictions = svmPredict(model, Xval);
    error = mean(double(predictions ~= yval));
    if(error < min_error(1))
      printf("C = %f and sigma = %f, is better. Error from %f down to %f", C_try, sigma_try, min_error(1), error);
      min_error = [error, C_try, sigma_try];
    endif
  endfor
endfor  


printf("Final C is %f, sigma is %f", min_error(2), min_error(3));

C = min_error(2);
sigma = min_error(3);


% =========================================================================

end
