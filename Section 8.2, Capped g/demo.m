clear; rng shuffle;
addpath(genpath('../library'))
%% Variables
n      = 1600;           % fixed sample size
bseq   = 0.15:0.5:5;     % increasing boundary
mustar = [1, 1];         % true mean
N      = 100;            % number of repetition for each n
cseq   = [0.1, 10, 100]; % values of slope c 
offset = [0, 0];         % position that the boundary expands from

%% Gaussian set up

% Variables
d     = 2;
sigma = 1;

% Log pdf
X    = sym('X',[d,1]);
mu   = sym('mu',[d*1,1]);
logp = log(exp(-sum((X-mu(1:2)).^2/sigma^2/2,1)));

% Auto-gradient of log pdf
gradX_logP  = gradient(logp,X);
gradX2_logP = diag(hessian(logp,X));
gradF       = matlabFunction(gradX_logP,'Vars',{X,mu});
grad2F      = matlabFunction(gradX2_logP,'Vars',{X,mu});
        
%% Loop over polygon boundary
error_polygon = zeros([length(bseq), length(cseq), N]);
pct_polygon   = zeros([length(bseq), length(cseq), N]);

for i = 1:length(bseq)
    b = bseq(i);
    
    % Construct square boundary
    square = [offset(1)-b, offset(2)-b; offset(1)-b, offset(2)+b;...
              offset(1)+b, offset(2)+b; offset(1)+b, offset(2)-b];  
          
    for j = 1:length(cseq)
        c = cseq(j);
        fprintf("b = %d, c = %d \n", b, c);
        parfor k = 1:N
            [est, pct] = fit_mvn_boundary(n, square, c, mustar, sigma, k, gradF, grad2F);
            error_polygon(i, j, k) = sqrt(sum((mustar' - est).^2, 1));
            pct_polygon(i, j, k)   = pct;
        end
    end
end
 
%% Loop over disjoint boundary
error_disjoint = zeros([length(bseq), length(cseq), N]);
pct_disjoint   = zeros([length(bseq), length(cseq), N]);

for i=1:length(bseq)
    b = bseq(i);
    
    % Construct disjoint boundaries
    window = [mustar(1)-b, mustar(2)-b-0.5; mustar(1)-b, mustar(2)-0.5; ...
              mustar(1)+b, mustar(2)-0.5;   mustar(1)+b, mustar(2)-0.5-b];
    window(:, :, 2) = [mustar(1)-b, mustar(2)+0.5; mustar(1)-b, mustar(2)+0.5+b;
              mustar(1)+b, mustar(2)+0.5+b; mustar(1)+b, mustar(2)+0.5];
        
    for j = 1:length(cseq)
        c = cseq(j);
        parfor k = 1:N
            [est, pct] = fit_mvn_boundary(n, window, c, mustar, sigma, k, gradF, grad2F);
            error_disjoint(i, j, k) = sqrt(sum((mustar' - est).^2, 1));
            pct_disjoint(i, j, k)   = pct;
        end
    end
    
end

%% Take means and standard deviation of arrays

% Polygon
polygon_mean = mean(error_polygon, 3);
polygon_std  = std(error_polygon, [], 3);
polygon_pct  = mean(pct_polygon, 3);

% Disjoint
disjoint_mean = mean(error_disjoint, 3);
disjoint_std  = std(error_disjoint, [], 3);
disjoint_pct  = mean(pct_disjoint, 3);

%% Get consistent y axis limits

% Need to take largest mean and largest mean + sd
y_up_poly     = max(polygon_mean, [], 'all');
y_up_poly_std = max(polygon_std + polygon_mean, [], 'all');

y_up_disj     = max(disjoint_mean, [], 'all');
y_up_disj_std = max(disjoint_std + disjoint_mean, [], 'all');

y_up = max([y_up_poly, y_up_poly_std, ...
    y_up_disj, y_up_disj_std]) + 0.1;
y_lo = 1e-2;

save('boundary')
%% Plot 2x2 grid
load boundary
figure()
t2 = tiledlayout(2, 2);
% nexttile
% hold on
leg = cell(1, length(cseq));
for i = 1:3
    leg{1,i} = strcat("L=", num2str(cseq(i)));
end

% Polygon
nexttile
hold on
for i = 1:3
    errorbar(bseq+i*.1, polygon_mean(:,i), polygon_std(:,i), ...
        '-s', 'MarkerSize', 6, 'linewidth',2);
end
legend(leg);
title(strcat("polygon boundary"))
xlabel("Boundary Size (Multiplier)");
h = ylabel("$\|\hat{\theta} - \theta^*\|$"); h.Interpreter = 'latex'
hold off
set(gca, 'YScale', 'Log')
ylim([y_lo y_up]); grid on;

% Polygon percent data capped 
nexttile
hold on
for i = 1:3
    plot(bseq, polygon_pct(:,i)*100, '-s', 'MarkerSize', 6, 'linewidth',2);
end
legend(leg,'location','southeast');
xlabel("Boundary Size");
h = ylabel("Ratio (\%) of $x$, $g(x)=1$"); h.Interpreter = 'latex';
hold off; grid on; 
ylim([0, 105]);

% Disjoint
nexttile
hold on
for i = 1:3
    errorbar(bseq+i*.1, disjoint_mean(:,i), disjoint_std(:,i), ...
        '-s', 'MarkerSize', 6, 'linewidth',2);
end
legend(leg);
title(strcat("disjoint boundary"))
xlabel("Boundary Size");
h = ylabel("$\|\hat{\theta} - \theta^*\|$"); h.Interpreter = 'latex';
set(gca, 'YScale', 'Log'); grid on; 
ylim([y_lo y_up]);

% Disjoint percent data capped 
nexttile
hold on
for i = 1:3
    plot(bseq, disjoint_pct(:,i)*100, '-s', 'MarkerSize', 6, 'linewidth',2);
end
legend(leg,'location','southeast');
xlabel("Boundary Size");
h = ylabel("Ratio (\%) of $x$, $g(x)=1$"); h.Interpreter = 'latex';
hold off; grid on;
ylim([0, 105]);

savefig("capped_g_benchmark.fig")


%% Plot example boundaries
figure;
b_plot_seq = [1,2,5];
t3 = tiledlayout(3, 2);

for i = 1:length(b_plot_seq)
    b      = b_plot_seq(i);
    % Construct square boundary
    square = [offset(1)-b, offset(2)-b; offset(1)-b, offset(2)+b;...
              offset(1)+b, offset(2)+b; offset(1)+b, offset(2)-b];  
          
    % Construct disjoint boundaries
    window = [mustar(1)-b, mustar(2)-b-0.5; mustar(1)-b, mustar(2)-0.5; ...
              mustar(1)+b, mustar(2)-0.5;   mustar(1)+b, mustar(2)-0.5-b];
    window(:, :, 2) = [mustar(1)-b, mustar(2)+0.5; mustar(1)-b, mustar(2)+0.5+b;
              mustar(1)+b, mustar(2)+0.5+b; mustar(1)+b, mustar(2)+0.5];
    

    nexttile
    plot_mvn(n, square, mustar, sigma, 1)
    title(strcat("b = ", num2str(b)))
    ylim([-2, 4]);
    xlim([-2, 4]);
    nexttile
    plot_mvn(n, window, mustar, sigma, 1)
    title(strcat("b = ", num2str(b)))
    ylim([-2, 4]);
    xlim([-2, 4]);
end

savefig("capped_g_boundary_example.fig")















