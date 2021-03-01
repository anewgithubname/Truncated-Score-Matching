clear; rng shuffle; 
addpath(genpath('../library'))
d = 2;
sigma = 1;

%% Symbolic differentiation
X = sym('X',[d,1]);
mu = sym('mu',[d*4,1]);
logp = log(exp(-sum((X-mu(1:2)).^2/sigma^2/2,1)) ...
        + exp(-sum((X-mu(3:4)).^2/2/sigma^2,1))...
        + exp(-sum((X-mu(5:6)).^2/2/sigma^2,1))...
        + exp(-sum((X-mu(7:8)).^2/2/sigma^2,1)));

gradX_logP = gradient(logp,X)
gradX2_logP = diag(hessian(logp,X))

gradF = matlabFunction(gradX_logP,'Vars',{X,mu});
grad2F = matlabFunction(gradX2_logP,'Vars',{X,mu});

%% Simulate Gaussian mixture data with four centres
n = 10000;
Xq = [randn(d,n/4)*sigma - 2, ...
      randn(d,n/4)*sigma + 2, ...
      mvnrnd([-2,2],eye(d)*sigma^2,n/4)', ...
      mvnrnd([2,-2],eye(d)*sigma^2,n/4)'];

% Create polygon for region V
Poly = [-.3 0; -.7 -.7; .2 -.7; .3 0; .5 .7; -.7 .7]*3;

% Get projected points onto the boundary
tic
[dist,t] = polydist(Xq, Poly);
toc

%% Run optimisation to find minima by truncSM
% initialize at the optimal position, with random perturbation added by randn.
x0 = [-2,-2, -2, 2, 2, 2, 2, -2]' + randn(d*4,1);

Xtrunc = Xq(:, dist<0); t = t(:, dist<0);
fprintf('effective sample size: %d \n',sum(dist<0))

tic
[mu_hat, fval] = fminunc(@(mu) truncSMMaha(mu, gradF, grad2F, Xtrunc, t, 1), x0);
toc

%% Run optimisation to find minima by RJ-MLE
X = sym('X',[d,1]);
mu = sym('mu',[d*4,1]);
xs = (rand(d,500000)-.5)*4.2;
[dist_xs,~] = polydist(xs,Poly);
xs = xs(:, dist_xs<0);

Z = @(mu,sigma) mean(exp(-sum((xs-mu).^2,1)/sigma^2/2),2);

logp = @(X, mu, sigma) log(exp(-sum((X-mu(1:2)).^2/sigma^2/2,1)) ...
         + exp(-sum((X-mu(3:4)).^2/sigma^2/2,1)) ...
         + exp(-sum((X-mu(5:6)).^2/sigma^2/2,1)) ...
         + exp(-sum((X-mu(7:8)).^2/sigma^2/2,1)))...
         - log(Z(mu(1:2), sigma) + Z(mu(3:4), sigma) + Z(mu(5:6), sigma) + Z(mu(7:8), sigma));
     
tic
mu_MLE = fminunc(@(mu) -mean(logp(Xtrunc, mu, 1),2), x0);
toc

%% Plot

% Data and polygon
hf = figure; hold on;
h1 = scatter(Xq(1,:),Xq(2,:),8,'k.'); grid on; 
h2 = fill(Poly(:,1),Poly(:,2),'g');
h2.FaceAlpha = .2; h2.EdgeColor = 'none';
h3 = scatter(Xtrunc(1,:), Xtrunc(2, :),4,'b');

% truncSM minima
h4 = scatter(mu_hat(1),mu_hat(2),72,'ro','MarkerFaceColor','r'); 
scatter(mu_hat(3),mu_hat(4),72,'ro','MarkerFaceColor','r');
scatter(mu_hat(5),mu_hat(6),72,'ro','MarkerFaceColor','r');
scatter(mu_hat(7),mu_hat(8),72,'ro','MarkerFaceColor','r');

% RJ-MLE minima
h5 = scatter(mu_MLE(1),mu_MLE(2),144,'g+', 'linewidth', 2);
scatter(mu_MLE(3),mu_MLE(4),144,'g+', 'linewidth', 2);
scatter(mu_MLE(5),mu_MLE(6),144,'g+', 'linewidth', 2);
scatter(mu_MLE(7),mu_MLE(8),144,'g+', 'linewidth', 2);

% Legend and save
legend([h2,h3,h4,h5],'V','X_q','Trunc-SM','RJ-MLE');
saveas(hf,sprintf('n_%d.png',n))
