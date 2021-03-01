clear; rng shuffle;
addpath('../library')

sTruncSM = zeros(36,5); sMLE = []; sMLEt = [];
nulist = .1:.1:.5;
parfor seed = 1:36
    for nu_id = 1:5
        
        nu = nulist(nu_id)
        tic
        %% Simulate samples
        data = mvnrnd(zeros(20,1), eye(20),10000);
        idx = randperm(size(data,1)); data = data(idx,:);

        inliear = data(1:200,:); data = data(201:end,:);
        % adding artificial outliers
        X = [inliear', mvnrnd(ones(size(data,2),1)*1, eye(size(data,2)),20)'];
        d = size(X,1);
        %% Fit one class SVM classifier for outliers using CVX
        y = ones(1,size(X,2));
        model = fitcsvm(X',y,'KernelScale','auto', 'OutlierFraction', nu);
        scale = model.KernelParameters.Scale;
        alpha = model.Alpha;
        sv = model.IsSupportVector;
        bias = model.Bias;
        f = @(x) exp(-comp_dist(x./scale, X(:,sv)./scale))*alpha + bias;
        
        % OSVM boundary function
        x = sym('x',  [d, 1], 'real');
        gPhi = gradient(f(x), x);
        gPhiF  = matlabFunction(gPhi,  'Vars', {x});
        
        % dataset for TruncSM
        X = [data(1:500,:)', mvnrnd(ones(size(data,2),1)*1, eye(size(data,2)),50)'];
        fX = f(X);
        
        %% Truncate outliers
        outliers = fX < 0;
        
        Xt = X(:, ~outliers);
        ntrunc = size(Xt, 2);
        
        %% TruncSM setup
        x     = sym('x',  [d, 1], 'real');
        mu    = sym('mu', [d, 1], 'real');
        sigma = sym('sigma', [d, 1], 'real');
        
        %why?
        logp  = -sum((x - mu).^2./(sigma.^2)/2, 1);
        
        gradX_logP  = gradient(logp, x);
        gradX2_logP = diag(hessian(logp, x));
        gradF  = matlabFunction(gradX_logP,  'Vars', {x, mu, sigma});
        grad2F = matlabFunction(gradX2_logP, 'Vars', {x, mu, sigma});
        
        %% compute g0, dg
        opts = optimset('fmincon');
        opts.MaxFunEvals = 100000;
        opts.MaxIter = 100000;
        opts.GradConstr = 'on';
        
        g  = zeros(1, ntrunc);
        Px = zeros(d, ntrunc);
        dg  = zeros(d, ntrunc);
        
        for i = 1:ntrunc
            
            [Px(:, i), ~, exitf] = fmincon(@(x) sum((x - Xt(:, i)).^2, 1), randn(d, 1), [], [], ...
                [], [], [], [], @(x) SVMcon(x, f, gPhiF), opts);
            if exitf < 0
                error('computing g error');
            end
            
            g(i) = norm(Xt(:, i) - Px(:,i));
            dg(:,i) = (Xt(:, i) - Px(:,i))./g(i);
            
            disp(sprintf('%.2f', i/ntrunc*100))
        end
        
        %% truncSM
        ini = [zeros(d, 1); ones(d,1)];
        
        est = fminunc(@(par) truncSMboth(par, gradF, grad2F, Xt, g, dg), ini, opts);
        toc
        
        %% compute holdout likelihood estimates
        MLEt = mean(Xt, 2);
        MLEt_sd = std(Xt,[],2);
        
        Xtest = data(501:end,:)';
        MLE = mean(X(:,1:500), 2);
        MLE_sd = std(X(:,1:500),[],2);
        
        LL = @(x,mu,Sigma) -sum((x-mu).^2./diag(Sigma).^2/2,1) - log(det(Sigma))/2 - log(2*pi)/2*d;
        LLtruncSM = mean(LL(Xtest, est(1:d), diag(est(end-d+1:end))),2)
        LLMLE = mean(LL(Xtest, MLE, diag(MLE_sd)),2)
        LLMLEt = mean(LL(Xtest, MLEt, diag(MLEt_sd)),2)
        
        sTruncSM(seed, nu_id) = LLtruncSM;
        sMLE(seed, nu_id) = LLMLE;
        sMLEt(seed, nu_id) = LLMLEt;
    end
end
%%
figure; errorbar(mean(sTruncSM,1),std(sTruncSM,[],1),'r')
hold on;
errorbar(mean(sMLE,1),std(sMLE,[],1),'k')
errorbar(mean(sMLEt,1),std(sMLEt,[],1),'b')
title(sprintf('mvn'))
save(sprintf('mvn'))

%% Functions
function [C, Ceq, gC, gCeq] = SVMcon(x, f, gPhiF)

C = [];
Ceq = f(x);
gC = [];
gCeq = gPhiF(x);

end

