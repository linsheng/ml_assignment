function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

cost = 0;
for i = 1:m
    h_theta = sigmoid(X(i, :) * (theta));
    cost = cost - y(i) * log(h_theta) - (1 - y(i)) * log(1 - h_theta);
end

b = 0;
for j = 1:length(theta)
    if (j > 1)
        b = b + theta(j) ^ 2;
    end
end

J = cost/m + (lambda*b)/(2*m);

for j = 1:length(theta)
    g = 0;
    for i = 1:m
        h_theta = sigmoid(X(i, :) * (theta));
        g = g + (h_theta - y(i)) * X(i, j);
    end
    if (j == 1)
        grad(j) = g/m;
    else
        grad(j) = g/m + lambda*theta(j)/m;
end

% =============================================================

end
