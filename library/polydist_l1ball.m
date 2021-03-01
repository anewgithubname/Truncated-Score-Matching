function [t,f] = polydist_l1ball(Xq, q)
% compute distance to a hemi l_q ball
% Xq: data
% q: the order of the norm

n = size(Xq, 2);
d = size(Xq, 1);
t = zeros(size(Xq,1), n);
f = zeros(1, size(Xq, 2));
% all combinations of -1 and 1, length d-1
all1s = [npermutek([-1,1],d-1)];
all1s = [all1s,ones(size(all1s,1),1)];
tt = repmat(NaN, [d, 2^(d-1) + 1]);
ff = repelem(NaN, 2^(d-1) + 1);
all1s = [all1s;[zeros(1,d-1),-1]];
bj= ones(size(all1s,1),1); bj(end) = 0;
 for i = 1:n
    opt = optimset('fmincon');
    opt.MaxIter = 10000;
    opt.MaxFunEvals = 10000;
    opt.Display = 'off';
    opt.GradObj = 'on';
%     opt.DerivativeCheck = 'on';
    obj = @(x) obj0(x, Xq(:,i), q);
    
    % optimise for all regions of the l_q ball
    for j = 1:size(all1s, 1)
        if q == 1
%             tic
%             x_xq = SolveBasisPursuitLp001(all1s(j,:), 1 - all1s(j,:)*Xq(:,i));
%             tt(:, j) = x_xq + Xq(:,i);
%             ff(j) = sum(abs(x_xq));
%             toc
%             tic
%             [tt(:, j), ff(j)] = fmincon(obj, zeros(size(Xq,1),1), [],[], all1s(j,:), 1, [],[], [], opt);
%             toc
              dist = abs((all1s(j,:)*Xq(:,i) - bj(j)) ./ all1s(j,:));
              [ff(j),idstar] = min(dist);
              
              tt(:, j) = -all1s(j,:)./abs(all1s(j,idstar));
        elseif q == 2
            z = Xq(:,i) - (all1s(j,:)*Xq(:,i) - bj(j))*all1s(j,:)'./(norm(all1s(j,:)).^2);
            tt(:, j) = (Xq(:,i) - z)./norm(Xq(:,i) - z);
            ff(j) = abs(all1s(j,:)*Xq(:,i) - bj(j))./ norm(all1s(j,:));
        else
%             [tt(:, j), ff(j)] = fmincon(obj, zeros(size(Xq,1),1), [],[], all1s(j,:), 1, [],[], [], opt);
        end
    end
%     [tt(:, j+1), ff(j+1)] = fmincon(obj, zeros(size(Xq,1),1), [],[], [zeros(1, d-1),1], ...
%             [0],[],[], [], opt);
        
    [minf, idx] = min(ff);
    f(i) = minf;
    t(:,i) = tt(:,idx);
end
end

function [f,g] = obj0(x,xq, q)
    if q == 1
        f = sum(abs(x-xq),1);
        g = sign(x-xq);
    else
        f = sqrt(sum(abs(x-xq).^2,1));
        g = 2*(x-xq);
    end
end