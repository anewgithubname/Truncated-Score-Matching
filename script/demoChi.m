clear; rng default; addpath(genpath('../'))
d = 2;
sigma = 0.06;

X = sym('X',[d,1]);
mu = sym('mu',[d*2,1]);

% define log p model, which is a Gaussian mixture
logp = log(exp(-sum((X-mu(1:2)).^2/sigma^2/2,1)) ...
        + exp(-sum((X-mu(3:4)).^2/sigma^2/2,1)));

% calculate the gradient and 2nd order gradient 
gradX_logP = gradient(logp,X);
gradX2_logP = diag(hessian(logp,X));

gradF = matlabFunction(gradX_logP,'Vars',{X,mu});
grad2F = matlabFunction(gradX2_logP,'Vars',{X,mu});

%% Load Chicago boundary and crime data
load data/chi.mat
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
hf = figure; 
h1 = scatter(Xq(1,:),Xq(2,:),8,'k.'); grid on; hold on;
h2 = fill(partialV(:,1),partialV(:,2),'g');
h2.FaceAlpha = .2; h2.EdgeColor = 'none';
h3 = scatter(Xq(1,dist<0), Xq(2, dist<0),4,'b');
sprintf('effective sample size: %d',sum(dist<0))

%% run the algorithm
% truncated SM
obj = @(mu) polyfit2(mu, gradF, grad2F, Xq, dist, t);
[mu_hat, fval] = fminunc(@(mu) polyfit2(mu, gradF, grad2F, Xq, dist, t),...
                                            randn(d*2,1)*.05 + [C;C])

% plot the mixture centers
h4 = scatter(mu_hat(1),mu_hat(2),72,'ro','MarkerFaceColor','r'); 
scatter(mu_hat(3),mu_hat(4),72,'ro','MarkerFaceColor','r');

% plot the 80% confidence interval
xs = sym('x',[2,1],'real');
h = fcontour((xs-mu_hat(1:2))'*inv(sigma^2*eye(2))*(xs-mu_hat(1:2)));
h.LevelList = 3.22; % the 80% percentile of chi-square with dof 2.
h = fcontour((xs-mu_hat(3:4))'*inv(sigma^2*eye(2))*(xs-mu_hat(3:4)));
h.LevelList = 3.22;

%MLE
% do the regular MLE, pretend the boundary does not exist
X = sym('X',[d,1]);
mu = sym('mu',[d*4,1]);
logp = log(exp(-sum((X-mu(1:2)).^2/sigma^2/2,1)) ...
         + exp(-sum((X-mu(3:4)).^2/sigma^2/2,1)));
f = matlabFunction(logp, 'Vars', {X,mu});
mu_MLE = fminunc(@(mu) -mean(f(Xq(:,dist<0), mu),2), randn(d*2,1)*.05 + [C;C])
h5 = scatter(mu_MLE(1),mu_MLE(2),72,'r+', 'linewidth', 1);
scatter(mu_MLE(3),mu_MLE(4),72,'r+', 'linewidth', 1);
axis([-87.95,-87.5, 41.6, 42.05])

legend([h2,h3,h4,h5],'V','X_q','Trunc-SM','MLE');
xlabel('latitude')
ylabel('longtitude')
saveas(hf,sprintf('n_%d.png',n))


%% plot g
[X1,X2] = meshgrid(-87.95:.01:-87.5, 41.6:.01:42.05);
Xt = [X1(:), X2(:)]';

[dist,t,tt] = polydist(Xt,partialV);
[~, g] = polyfit2(mu_hat, gradF, grad2F, Xt, dist, t);
Z = zeros(1,size(Xt,2));
Z(dist<0) = g;
hg0 = figure; 
h = surf(X1,X2,reshape(Z,size(X1)));
view(0,90)
shading interp
lightangle(-45,30)
h.FaceLighting = 'gouraud';
h.AmbientStrength = 0.5;
h.DiffuseStrength = 0.8;
h.SpecularStrength = 0.9;
h.SpecularExponent = 25;
h.BackFaceLighting = 'unlit';
saveas(hg0,'gChi.png')