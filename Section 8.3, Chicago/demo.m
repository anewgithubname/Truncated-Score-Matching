clear; rng shuffle;
addpath(genpath('../data'))
d = 2;
sigma = 0.06;

%% Symbolic differentiation
X = sym('X',[d,1]);
mu = sym('mu',[d*2,1]);

% define log p model, which is a Gaussian mixture with two centers
logp = log(exp(-sum((X-mu(1:2)).^2/sigma^2/2,1)) ...
    + exp(-sum((X-mu(3:4)).^2/sigma^2/2,1)));

% calculate the gradient and 2nd order gradient
gradX_logP = gradient(logp,X);
gradX2_logP = diag(hessian(logp,X));

gradF = matlabFunction(gradX_logP,'Vars',{X,mu});
grad2F = matlabFunction(gradX2_logP,'Vars',{X,mu});

%% Load Chicago boundary and crime data
load chi.mat
SX = SX(~isnan(SX));
SY = SY(~isnan(SY));
partialV = [SX(1:11350);SY(1:11350)]';

% geometric center of the polygon
C = mean(partialV,1)';

id = find(PrimaryType == 'HOMICIDE');
Xq = [Longitude(id),Latitude(id)]';

% calculate the distance to boundary and the projection t on the boundary for
% all datas points in Xq
[dist,t] = polydist(Xq,partialV);

% plot the data and Chicago area
n = size(Xq,2);

% RJ sampling
X = sym('X',[d,1]);
mu = sym('mu',[d*2,1]);
xs = randn(d,5000).*std(Xq,[],2)*5+mean(Xq,2);
[dist_xs,~] = polydist(xs,partialV);
xs = xs(:, dist_xs<0);

list_SM = zeros(4,500);
list_MLE = zeros(4,500);

parfor n = 1:500
    %% run the algorithm
    % truncated SM
    x0 = randn(d*2,1)*.06 + [mean(Xq,2);mean(Xq,2)];
    tic
    [mu_hat, fval] = fminunc(@(mu) truncSMMaha(mu, gradF, grad2F, Xq,  t, 1), x0)
    toc
    list_SM(:,n) = mu_hat;
    
    %MLE
    Z = @(mu,sigma) mean(exp(-sum((xs-mu).^2,1)/sigma^2/2),2);
    % Z = @(mu,sigma) 1;
    
    logp = @(X, mu, sigma) log(exp(-sum((X-mu(1:2)).^2/sigma^2/2,1)) ...
        + exp(-sum((X-mu(3:4)).^2/sigma^2/2,1)))...
        - log(Z(mu(1:2), sigma) + Z(mu(3:4), sigma));
    
    tic
    mu_MLE = fminunc(@(mu) -mean(logp(Xq, mu, sigma),2), x0)
    toc
    list_MLE(:,n) = mu_MLE;
end

%% Plot
hf = figure; hold on;

bandwidth = sigma *2;
% plot the mixture centers
for i = 1:500
    h4 = rectangle('Position',[list_SM(1,i)-bandwidth/2, list_SM(2,i)-bandwidth/2, bandwidth, bandwidth],'Curvature',[1 1]);
    h4.FaceColor = [1,0,0,.002];
    h4.EdgeColor = 'none';
    hold on;
    h = rectangle('Position',[list_SM(3,i)-bandwidth/2, list_SM(4,i)-bandwidth/2, bandwidth, bandwidth],'Curvature',[1 1]);
    h.FaceColor = [1,0,0,.002];
    h.EdgeColor = 'none';
end


for i = 1:500
    h5 = rectangle('Position',[list_MLE(1,i)-bandwidth/2, list_MLE(2,i)-bandwidth/2, bandwidth, bandwidth],'Curvature',[1 1]);
    h5.FaceColor = [0,0,1,.002];
    h5.EdgeColor = 'none';
    hold on;
    h = rectangle('Position',[list_MLE(3,i)-bandwidth/2, list_MLE(4,i)-bandwidth/2, bandwidth, bandwidth],'Curvature',[1 1]);
    h.FaceColor = [0,0,1,.002];
    h.EdgeColor = 'none';
end

h3 = scatter(Xq(1,:),Xq(2,:),8,'b'); grid on; hold on;
h2 = fill(partialV(:,1),partialV(:,2),'g');
h2.FaceAlpha = .2; h2.EdgeColor = 'none';
sprintf('effective sample size: %d',sum(dist<0))

h4 = scatter((list_SM(1,10)),(list_SM(2,10)),72,'ro','MarkerFaceColor','r');
scatter((list_SM(3,10)), (list_SM(4,10)),72,'ro','MarkerFaceColor','r');

h5 = scatter((list_MLE(1,10)),(list_MLE(2,10)),72,'b+', 'linewidth', 3);
scatter((list_MLE(3,10)), (list_MLE(4,10)),72,'b+', 'linewidth', 3);

% plot the 80% confidence interval
xs = sym('x',[2,1],'real');
h = fcontour((xs-list_SM(1:2,10))'*inv(sigma^2*eye(2))*(xs-list_SM(1:2,10)));
h.LevelList = 3.22; % the 80% percentile of chi-square with dof 2.
h.LineColor = 'r'; h.LineWidth = 2; h.LineStyle = '--';
h = fcontour((xs-list_SM(3:4,10))'*inv(sigma^2*eye(2))*(xs-list_SM(3:4,10)));
h.LevelList = 3.22;
h.LineColor = 'r'; h.LineWidth = 2; h.LineStyle = '--';

xs = sym('x',[2,1],'real');
h = fcontour((xs-list_MLE(1:2,10))'*inv(sigma^2*eye(2))*(xs-list_MLE(1:2,10)));
h.LevelList = 3.22; % the 80% percentile of chi-square with dof 2.
h.LineColor = 'b'; h.LineWidth = 2; h.LineStyle = '--';
h = fcontour((xs-list_MLE(3:4,10))'*inv(sigma^2*eye(2))*(xs-list_MLE(3:4,10)));
h.LevelList = 3.22;
h.LineColor = 'b'; h.LineWidth = 2; h.LineStyle = '--';

X = sym('X',[d,1]);
mu = sym('mu',[d*4,1]);
logp = log(exp(-sum((X-mu(1:2)).^2/sigma^2/2,1)) ...
         + exp(-sum((X-mu(3:4)).^2/sigma^2/2,1)));
f = matlabFunction(logp, 'Vars', {X,mu});
x0 = randn(d*2,1)*.06 + [mean(Xq,2);mean(Xq,2)];
mu_MLE = fminunc(@(mu) -mean(f(Xq, mu),2), x0);

h6 = scatter((mu_MLE(1,:)),(mu_MLE(2,:)),72,'ko', 'linewidth', 3);
scatter((mu_MLE(3,:)), (mu_MLE(4,:)),72,'ko', 'linewidth', 3);

axis([-87.95,-87.5, 41.6, 42.05])
%
legend([h2,h3, h4, h5, h6],'V','X_q','TruncSM', 'RJ-MLE', 'MLE');
xlabel('latitude')
ylabel('longtitude')
saveas(hf,sprintf('n_%d.png',n))


% %% plot g
% [X1,X2] = meshgrid(-87.95:.001:-87.5, 41.6:.001:42.05);
% Xt = [X1(:), X2(:)]';
% 
% [dist,t,tt] = polydist(Xt,partialV);
% [~, g] = truncSM(mu_hat, gradF, grad2F, Xt, dist, t);
% Z = zeros(1,size(Xt,2));
% Z(dist<0) = g;
% hg0 = figure;
% h = surf(X1,X2,reshape(Z,size(X1)));
% view(0,90)
% shading interp
% lightangle(-45,30)
% h.FaceLighting = 'gouraud';
% h.AmbientStrength = 0.5;
% h.DiffuseStrength = 0.8;
% h.SpecularStrength = 0.9;
% h.SpecularExponent = 25;
% h.BackFaceLighting = 'unlit';
% saveas(hg0,'gChi.png')