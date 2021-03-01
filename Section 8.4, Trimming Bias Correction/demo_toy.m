clear; rng default;
addpath('../library')

%% Options
d     = 2;
nu    = 0.2;   % Proportion of outliers in OSVM
n_in  = 500;   % number of inlier
n_out = 50;    % number of outlier

inliear_mu    = [0, 0]; %inlier mean
outlier_mu = [3, 3.5]; %outlier mean

inliear_sd    = diag([1.5, 1]); %inliear std
outlier_sd = 0.8; %outlier std

%% Simulate in-liers
% generate inliear and outlier 
X_in  = mvnrnd(inliear_mu,    inliear_sd.^2,    n_in)';
X_out = [mvnrnd(outlier_mu, eye(d)*outlier_sd.^2, floor(n_out))'];
X = [X_in, X_out];

idx = randperm(size(X,2)); X = X(:,idx);
Xb = X;

%% Fit one class SVM classifier for outliers using CVX

PhiX = Phi(X,Xb);

p = size(PhiX,1);        
n = size(X, 2);    
cvx_begin
variables w(p) xi(n) rho(1)
minimize( .1*w'*w + (1/(nu*n))*sum(xi) - rho )
subject to
% no for loop.
xi >= 0;
w' * PhiX >= rho - xi';
cvx_end

% decision function
f = w'*PhiX - rho;

x = sym('x',  [d, 1]);
gPhi = gradient(w' * Phi(x,Xb)- rho, x);
gPhiF  = matlabFunction(gPhi,  'Vars', {x});
%% Truncate to within region
outliers = f < 0;

Xt = X(:, ~outliers);
ntrunc = size(Xt, 2);

%% TruncSM setup
x     = sym('x',  [d, 1]);
mu    = sym('mu', [d, 1]);
sigma = sym('sigma', [d, 1]);

logp  = -sum((x - mu).^2./(sigma.^2)/2, 1);

gradX_logP  = gradient(logp, x);
gradX2_logP = diag(hessian(logp, x));
gradF  = matlabFunction(gradX_logP,  'Vars', {x, mu, sigma});
grad2F = matlabFunction(gradX2_logP, 'Vars', {x, mu, sigma});

%% compute g0, dg
opts = optimset('fmincon');
opts.MaxFunEvals = 100000;
opts.MaxIter = 100000;
g  = zeros(1, ntrunc);
Px = zeros(d, ntrunc);
dg  = zeros(d, ntrunc);

parfor i = 1:ntrunc   
    
    [Px(:, i), ~, exitf] = fmincon(@(x) sqrt(sum((x - Xt(:, i)).^2, 1)), randn(d, 1), [], [], ...
        [], [], [], [], @(x) SVMcon(x,Xb, w, rho, gPhiF), opts);
    if exitf <= 0
        error('computing g error');
    end
       
    g(i) = norm(Xt(:, i) - Px(:,i));
    dg(:,i) = (Xt(:, i) - Px(:,i))./g(i);
    
    i/ntrunc*100
end

%% truncSM
ini = [zeros(d, 1); ones(d, 1)];

est = fminunc(@(par) truncSMboth(par, gradF, grad2F, Xt, g, dg), ini, opts);

%% Compare std estimates
MLEt_sd = std(Xt,[],2);

fprintf("True SD: [%3.4f, %3.4f] \n", inliear_sd(1,1), inliear_sd(2,2));
fprintf("truncSM SD: [%3.4f, %3.4f] \n", est(3), est(4));
fprintf("MLE SD: [%3.4f, %3.4f] \n", MLEt_sd(1), MLEt_sd(2));

%% Plot estimated confidence regions. 
MLE = mean(X, 2);
MLEt = mean(Xt, 2);
MLE_sd = std(X,[],2);
MLEt_sd = std(Xt,[],2);

figure;
hold on
scatter(X(1, :), X(2, :), 100, 'k.', 'DisplayName', '$X_q$')
scatter(Xt(1, :), Xt(2, :), 100, 'b.', 'DisplayName', 'Selected inliers')
h = fcontour(@(x1,x2) w'*Phi([x1;x2], Xb) - rho);
h.LevelList = [0];
h.LineColor = 'k';
h.LineStyle = '--';
h.LineWidth = 2;
h.DisplayName = '$V$ determined by OSVM';

Sigma_TruncSM = diag(est(3:4).^2);
h = fcontour(@(x1,x2) sqrt(([x1; x2] - inliear_mu')'*inv(inliear_sd.^2)*([x1; x2] - inliear_mu')));
h.LevelList = [sqrt(6)];
h.LineColor = 'b';
h.LineWidth = 2;
h.DisplayName = '95\% confidence region (true)';

Sigma_MLEt= diag(MLEt_sd.^2);
h = fcontour(@(x1,x2) sqrt(([x1; x2] - MLEt)'*inv(Sigma_MLEt)*([x1; x2] - MLEt)));
h.LevelList = [sqrt(6)];
h.LineColor = 'g';
h.LineWidth = 2;
h.DisplayName = 'Est. 95\% confidence region (MLE)'

Sigma_TruncSM = diag(est(3:4).^2);
h = fcontour(@(x1,x2) sqrt(([x1; x2] - est(1:2))'*inv(Sigma_TruncSM)*([x1; x2] - est(1:2))));
h.LevelList = [sqrt(6)];
h.LineColor = 'r';
h.LineWidth = 2;
h.DisplayName = 'Est. 95\% confidence region (TruncSM)';


%% output likelihood
Xtest = mvnrnd(inliear_mu,    inliear_sd.^2,    100000)';

LL = @(x,mu,Sigma) -sum((x-mu).^2./diag(Sigma).^2/2,1) - log(det(Sigma))/2 - log(2*pi)/2*d;
LLtruncSM = mean(LL(Xtest, est(1:d), diag(est(end-d+1:end))),2)
LLMLE = mean(LL(Xtest, MLE, diag(MLE_sd)),2)
LLMLEt = mean(LL(Xtest, MLEt, diag(MLEt_sd)),2)

%% OSVM boundary
function [C, Ceq] = SVMcon(x, xb, w,  rho, gPhiF)

C = [];
Ceq = w' * Phi(x,xb) - rho;
gC = [];
gCeq = gPhiF(x);
end

