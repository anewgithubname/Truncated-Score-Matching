function out = Phi(x, xb)
out = ones(1, size(x,2));
dist2 = comp_dist(xb,x);
out = [out; kernel_gau(dist2,2.14)];

% out = [out; x; x.^2];

end
