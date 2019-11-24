function [f, gXq, dgXq] = polyfit2(theta, gradF, grad2F, Xq, dist, Px)
% compute the trunc SM objective function
% theta: parameter of your log density function
% gradF: the gradient of log density function
% grad2F: the 2nd order gradient of log density function
% Xq: dataset, d times n
% d: distance to boundary
% Px: projection of Xq to boundary, same shape as Xq

Xq = Xq(:, dist<0);
Px = Px(:,dist<0);

% weak gradient of g_0 at Xq
dgXq = (Xq- Px)./sqrt(sum((Xq- Px).^2,1));
% g_0(Xq)
gXq = sqrt(sum((Xq- Px).^2,1));

% construnct the trunc-SM objective function
t1 = mean(sum((gradF(Xq,theta)).^2.*gXq,1),2);

t2 = mean(sum(grad2F(Xq,theta),1).*gXq,2) + ... 
     mean(sum(gradF(Xq,theta).*dgXq,1),2);

f = t1 + 2*t2;

end