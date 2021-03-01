function [est, pct_capped] = fit_mvn_boundary(n, Poly, c, mustar, sigma, seed, gradF, grad2F)

rng(seed); 

%% Simulate data
Sigma = [sigma, 0; 0, sigma];
Xdat =  mvnrnd(mustar, Sigma, n)';

%% Collect any disjoint boundaries
disjoints = size(Poly, 3);

if disjoints > 1
    g_min = zeros(size(Xdat, 2), disjoints);
    t_min = zeros(size(Xdat, 1), size(Xdat, 2), disjoints);
    for i = 1:disjoints
        Poly_i = Poly(:,:,i);
        [g_i, t_i] = polydist(Xdat, Poly_i);
        g_min(:,i) = g_i;
        t_min(:, :, i) = t_i;
    end

    [g, g_ind] = min(g_min, [], 2);
    g = g';
    trunc = g < 0;
    t = NaN(size(Xdat));
    for i = 1:disjoints
        t(:,g_ind==i) = t_min(:,g_ind==i,i);     
    end    
else 
    [g, t] = polydist(Xdat, Poly);
    trunc  = g < 0;
end

%% Truncate outside boundary
Xtrunc = Xdat(:, trunc);
g2     = abs(g(trunc));
t2     = t(:, trunc);

% 'Clip' g to be maximum of length U=1
g2 = min(c.*abs(g2), 1);

% Gradient of g at truncated points
dg = c.*(Xtrunc - t2)./sqrt(sum((Xtrunc - t2).^2, 1));
dg(:, g2==1.000) = 0;
pct_capped = sum(g2==1.000)/size(Xtrunc, 2);

%% Optimise TruncSM

% Make wrapper
obj = @(theta) truncSM(theta, gradF, grad2F, Xtrunc, g2, dg);
est = fminunc(obj, randn(2, 1), optimoptions('fminunc','Display', 'off'));





























end