function [f, gXq, dgXq] = truncSMMaha(theta, gradF, grad2F, Xq, Px, COV)
% compute the trunc SM objective function with Mahalanobis distance
% function g
% theta: parameter of your log density function
% gradF: the gradient of log density function
% grad2F: the 2nd order gradient of log density function
% Xq: dataset, d times n
% Px: projection of Xq to boundary, same shape as Xq
% COV: covariance of the mahalanobis distance. COV = 1 <=> Euclidean
% distance.

% g_0(Xq)
gXq = sqrt(sum((chol(inv(COV))*(Xq - Px)).^2,1));

% weak gradient of g_0 at Xq
dgXq = inv(COV)*(Xq - Px)./sqrt(sum((chol(inv(COV))*(Xq - Px)).^2,1));

% construnct the trunc-SM objective function
t1 = mean(sum((gradF(Xq,theta)).^2.*gXq,1),2);
t2 = mean(sum(grad2F(Xq,theta),1).*gXq,2) + ...
    mean(sum(gradF(Xq,theta).*dgXq,1),2);

f = t1 + 2*t2;
end