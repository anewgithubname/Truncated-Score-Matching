function plot_mvn(n, Poly, mustar, sigma, seed)

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
t2     = t(:, trunc);

%% Plot
scatter(Xdat(1,:), Xdat(2,:), 2, 'k', 'MarkerFaceColor','k')
hold on
scatter(Xtrunc(1,:), Xtrunc(2,:), 2, 'blue', 'MarkerFaceColor','blue')
% for i = 1:disjoints
%     scatter(Poly(:,1,1), Poly(:,2,1), 2, 'green', 'filled', 'MarkerFaceColor', 'green')
% end
% scatter(t2(1,:), t2(2,:), 2, 'red', 'MarkerFaceColor', 'red')
for i = 1:disjoints
h2 = fill(Poly(:,1,i),Poly(:,2,i),'g');
h2.FaceAlpha = .2; h2.EdgeColor = 'none';
end
end