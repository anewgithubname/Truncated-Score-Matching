function f = truncSMboth(theta, gradF, grad2F, Xq, g, dg)
% compute the trunc SM objective function with a pre-supplied distance
% function g and gradient dg
% theta: parameter of your log density function
% gradF: the gradient of log density function
% grad2F: the 2nd order gradient of log density function
% Xq: dataset, d times n
% d: distance to boundary
% Px: projection of Xq to boundary, same shape as Xq

% Estimating both mean mu and sigma (diagonal of cov matrix, length d)

d = size(Xq, 1);

% construnct the trunc-SM objective function
t1 = mean(sum(gradF(Xq, theta(1:d), theta((d+1):(2*d))).^2.*g, 1), 2);
t2 = mean(sum(grad2F(Xq, theta(1:d), theta((d+1):(2*d))).*g,  1), 2) + ... 
     mean(sum(gradF(Xq, theta(1:d), theta((d+1):(2*d))).*dg,  1), 2);

f = t1 + 2*t2;

if any(theta((d+1):(2*d)) < 0)
    f = 9e10;
end

end