function [d,t,tt] = polydist(Xq,Poly)
n = size(Xq,2);
d = zeros(1,n);
t = zeros(2,n);
tt = zeros(2,n);

parfor i = 1:n
    [d(i),t1,t2] = p_poly_dist(Xq(1,i), Xq(2,i), Poly(:,1), Poly(:,2));
    t(:,i) = [t1;t2];
end
end