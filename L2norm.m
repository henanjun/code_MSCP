function [featureNorm] = L2norm(feature)
% Description: Each feature dimension to has the same L2 norm
% input: feature each row is a feature attribute;

[m,n] = size(feature);
aul = sqrt(sum(feature.^2,2));

featureNorm = feature./repmat(aul,1,n);


end