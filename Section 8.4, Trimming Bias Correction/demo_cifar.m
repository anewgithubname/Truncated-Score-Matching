clear; rng shuffle;
addpath('../library')
addpath('../data/cifar10')

d = 10;

for digit = [0:9]
    % find PCA proejction matrix
    load test_batch.mat
    proj = pca(double(data(labels==digit,:)),...
                'VariableWeights','variance', 'NumComponents', d);
    % load data for training OSVM
    load data_batch_1.mat
    data_OSVM = double(data(labels==digit,:))*proj;
    data_OSVM = data_OSVM - mean(data_OSVM,1);
    data_OSVM = data_OSVM ./ std(data_OSVM);
    
    % load data for TruncSM and testing
    load data_batch_2.mat
    data_truncSM = double(data(labels==digit,:))*proj;
    data_truncSM = data_truncSM - mean(data_truncSM,1);
    data_truncSM = data_truncSM ./ std(data_truncSM);
    
    sTruncSM = []; sMLE = []; sMLEt = [];
    nulist = .1:.1:.5;
    for nu_id = 1:length(nulist)
        nu = nulist(nu_id)
        
        % create outlier truncation boundary using OSVM
        % load inlier
        data_OSVM = data_OSVM(1:200,:);
        
        % add outlier
        X = [data_OSVM', mvnrnd(ones(size(data_OSVM,2),1)*1, eye(size(data_OSVM,2)),20)'];
        y = ones(1,size(X,2));
        d = size(X,1);
        
        % OSVM
        model = fitcsvm(X',y,'KernelScale','auto', 'OutlierFraction', nu);
        % get outlier decision function
        scale = model.KernelParameters.Scale;
        alpha = model.Alpha;
        sv = model.IsSupportVector;
        bias = model.Bias;
        
        x = sym('x',  [d, 1], 'real');
        f = @(x) exp(-comp_dist(x./scale, X(:,sv)./scale))*alpha + bias;
        % get gradient of prediction function
        gPhi = gradient(f(x), x);
        gPhiF  = matlabFunction(gPhi,  'Vars', {x});
        
        tic
        parfor seed = 1:72
            %% load in-liers
            % randomize data
            idx = randperm(size(data_truncSM,1)); score = data_truncSM(idx,:);
            
            
            %% Truncate to within region
            % add outlier
            X = [score(1:500,:)', mvnrnd(ones(size(score,2),1)*1, eye(size(score,2)),50)'];
            fX = f(X);
            outliers = fX < 0;
            % remove outlier
            Xt = X(:, ~outliers);
            ntrunc = size(Xt, 2);
            
            %% TruncSM setup
            x     = sym('x',  [d, 1], 'real');
            mu    = sym('mu', [d, 1], 'real');
            sigma = sym('sigma', [d, 1], 'real');
            
            logp  = -sum((x - mu).^2./(sigma.^2)/2, 1);
            
            gradX_logP  = gradient(logp, x);
            gradX2_logP = diag(hessian(logp, x));
            gradF  = matlabFunction(gradX_logP,  'Vars', {x, mu, sigma});
            grad2F = matlabFunction(gradX2_logP, 'Vars', {x, mu, sigma});
            
            %% find g_0, dg0
            opts = optimset('fmincon');
            opts.MaxFunEvals = 100000;
            opts.MaxIter = 100000;
            opts.GradConstr = 'on';
            opts.Display = 'none';
            
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
                
            end
            
            %% truncSM
            ini = [zeros(d, 1); ones(d,1)];
            est = fminunc(@(par) truncSMboth(par, gradF, grad2F, Xt, g, dg), ini, opts);
            
            %% compute holdout likelihood estimates
            % MLE estimates
            MLEt = mean(Xt, 2);
            MLEt_sd = std(Xt,[],2);

            % Ground truth
            Xtest = score(501:end,:)';
            MLE = mean(X(:,1:500), 2);
            MLE_sd = std(X(:,1:500),[],2);
            
            % compute hold out likelihood
            LL = @(x,mu,Sigma) -sum((x-mu).^2./diag(Sigma).^2/2,1) - log(det(Sigma))/2 - log(2*pi)/2*d;
            LLtruncSM = mean(LL(Xtest, est(1:d), diag(est(end-d+1:end))),2)
            LLMLE = mean(LL(Xtest, MLE, diag(MLE_sd)),2)
            LLMLEt = mean(LL(Xtest, MLEt, diag(MLEt_sd)),2)
            
            sTruncSM(seed, nu_id) = LLtruncSM;
            sMLE(seed, nu_id) = LLMLE;
            sMLEt(seed, nu_id) = LLMLEt;
        end
        toc
    end
    save(sprintf('res%d', digit))
end

%% OSVM prediction function
function [C, Ceq, gC, gCeq] = SVMcon(x, f, gPhiF)

C = [];
Ceq = f(x);
gC = [];
gCeq = gPhiF(x);

end

