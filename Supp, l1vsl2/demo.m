clear; 
addpath('../library')

%% Set up variables
sigma = 1;

nTrial = 400; % number of repetitions
n_list = 150;  % number of samples
d_list = 2:5; % dimensionalities 


%% Embedded Loops for different variables
s = zeros(length(d_list),2,length(n_list),nTrial);
%%
tic
for k = 1:length(d_list)
    
    
    %% Symbolic differentiation
    d = d_list(k);
    
    X = sym('X',[d,1]);
    mu = sym('mu',[d*1,1]);
    logp = log(exp(-sum((X-mu(1:d)).^2/sigma^2/2,1)));
    gradX_logP = gradient(logp,X);
    gradX2_logP = diag(hessian(logp,X));
    gradF = matlabFunction(gradX_logP,'Vars',{X,mu});
    grad2F = matlabFunction(gradX2_logP,'Vars',{X,mu});


    fprintf("|| k = %d, d = %d \n", k, d);
    
    for i = 1:length(n_list)
        
        n = n_list(i);
        fprintf("i = %d, n = %d \n", i, n);
        
%         tic
        parfor seed = 1:nTrial

            % Truncate according to an 'hemi-L1 ball'
            Xq = [];
            rng(seed);
            
            % prepare Xq by rejection sampling
            for iter=1:10000
                Xqt =  mvnrnd(ones(d,1)*.5, eye(d), 100000)';
                Xq = [Xq, Xqt(:, sum(abs(Xqt),1) < 1 & Xqt(d,:) > 0)];
                if size(Xq,2) > n
                    Xq = Xq(:, 1:n);
                    break;
                end
            end
            if iter == 10000
                error('cannot sample')
            end
            [dgXq1,gXq1] = polydist_l1ball(Xq, 1);
            [dgXq2,gXq2] = polydist_l1ball(Xq, 2);

            [mu_hat_l1, fval1] = fminunc(@(mu) truncSM(mu, gradF, grad2F, Xq, gXq1, dgXq1), zeros(d,1));
            [mu_hat_l2, fval2] = fminunc(@(mu) truncSM(mu, gradF, grad2F, Xq, gXq2, dgXq2), zeros(d,1));
            
            s(k, :, i, seed) = [norm(mu_hat_l1-(.5));  norm(mu_hat_l2-(.5))];
        end
%         toc
    end
end
toc
save('l1l2')

%% Plot
load l1l2
figure(); hold on

% L1 norm
si = s(:, 1, :, :);
msi = (mean(si, 4));
ssi = (std(si, [], 4));
errorbar(d_list, msi, ssi, 'Linewidth', 2, 'Marker', 'o')

% L2 norm
si = s(:, 2, :, :);
msi = (mean(si, 4));
ssi = (std(si, [], 4));
errorbar(d_list + .05, msi, ssi,'Linewidth', 2, 'Marker', 'o')

xlabel('d');
ylabel('$\| \theta^{*} - \hat{\theta}\|$','interpreter','latex');
h = legend(["$d(x,y) = \|x-y\|_1$", "$d(x,y) = \|x-y\|_2$", "coordinate"]);
h.Interpreter = 'latex';
grid on

