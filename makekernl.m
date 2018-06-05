function [kernel_matrix] = makekernl(X1,X2,param)      

switch param.kernel
    
            
            case('Log-E inner')
             kernel_matrix =  (X1*X2');
            case('Log-E poly.')
            % The scripts that run fast
            X1 = bsxfun(@rdivide, X1, sqrt(sum(X1.^2, 2)));
            X2 = bsxfun(@rdivide, X2, sqrt(sum(X2.^2, 2)));         
            if isequal(X1, X2)
                kernel_matrix = 1 - squareform(pdist(X1, 'cosine'));
            else
                kernel_matrix = 1 - pdist2(X1, X2, 'cosine');
            end
            kernel_matrix =  kernel_matrix .^ param.n;
            
        case('Log-E exp.')
            % The scripts that run fast
            X1 = bsxfun(@rdivide, X1, sqrt(sum(X1.^2, 2)));
            X2 = bsxfun(@rdivide, X2, sqrt(sum(X2.^2, 2)));
            if isequal(X1, X2)
                kernel_matrix = 1 - squareform(pdist(X1, 'cosine'));
            else
                kernel_matrix = 1 - pdist2(X1, X2, 'cosine');
            end
            kernel_matrix =  exp(kernel_matrix .^ param.n);


        case('Log-E Gauss.')
           % The scripts that run fast
             if isequal(X1, X2)
                kernel_matrix = squareform(pdist(X1, 'euclidean'));
            else
                kernel_matrix = pdist2(X1, X2, 'euclidean');
             end
            if isfield(param,'Beta')
                Beta =param.Beta;
            else
                Beta = 1/mean(mean(kernel_matrix));
            end
             kernel_matrix =  exp(-Beta.*kernel_matrix.^2);
             
end
end