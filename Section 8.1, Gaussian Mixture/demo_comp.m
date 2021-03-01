clear; rng shuffle;
addpath(genpath('../library'))
d = 2;
sigma = 1;
n = 10000;
Poly = [-.3 0; -.7 -.7; .2 -.7; .3 0; .5 .7; -.7 .7]*3;

%% Symbolic Differentiation
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

%% for loop over different number of rejection samples
for nsample = [500,1000,2000,4000,8000,12000]
    fprintf("nsample = %d \n", nsample);
    for iter = 1:96
        
        %% Simulate dataset and set up truncated points
        Xq = [randn(d,n/4)*sigma - 2, ...
            randn(d,n/4)*sigma + 2, ...
            mvnrnd([-2,2],eye(d)*sigma^2,n/4)', ...
            mvnrnd([2,-2],eye(d)*sigma^2,n/4)'];
        
        
        % initialization point
        x0 = [2,-2,-2,-2,-2,2,2,2]';
        [dist,~] = polydist(Xq,Poly);
        
        % get truncated points. Do not count in the timing. 
        Xq = Xq(:, dist<0);

        %% TruncSM, we run it for every nsample.
        tic
        [dist,t] = polydist(Xq,Poly);
        fprintf('iter = %d, effective sample size: %d \n', iter, sum(dist<0));
           
        [mu_hat, fval] = fminunc(@(mu) truncSMMaha(mu, gradF, grad2F, Xq, t,1), x0, ...
                                 optimoptions('fminunc', 'Display', 'off'));
        time1(iter) = toc;
        t1 = norm(mu_hat - [2,-2,-2,-2,-2,2,2,2]')
        e1(iter) = t1;
        
        tic
        
        %% RJ-MLE
        xs = (rand(d,nsample)-.5)*4.2;
        [dist_xs,~] = polydist(xs,Poly);
        xs = xs(:, dist_xs<0);
        
        Z = @(mu,sigma) mean(exp(-sum((xs-mu).^2,1)/sigma^2/2),2);
        
        logp = @(X, mu, sigma) log(exp(-sum((X-mu(1:2)).^2/sigma^2/2,1)) ...
            + exp(-sum((X-mu(3:4)).^2/sigma^2/2,1)) ...
            + exp(-sum((X-mu(5:6)).^2/sigma^2/2,1)) ...
            + exp(-sum((X-mu(7:8)).^2/sigma^2/2,1)))...
            - log(Z(mu(1:2), sigma) + Z(mu(3:4), sigma) + Z(mu(5:6), sigma) + Z(mu(7:8), sigma));
        
        mu_MLE = fminunc(@(mu) -mean(logp(Xq, mu, 1),2), x0, ...
                         optimoptions('fminunc', 'Display', 'off'))
        
        time2(iter) = toc;
        t2 = norm(mu_MLE - [2,-2,-2,-2,-2,2,2,2]');
        e2(iter) = t2;
    end
    save(sprintf('SM1'),'e1', 'time1')
    save(sprintf('RJ-MLE%d',nsample),'e2', 'time2')
end
