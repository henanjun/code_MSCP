function [convfeacov] = calc_stackedCovdescriptor(rt_img_dir,net,conv_idex, data_set_name,img_type, reduced_fc_num)
%==========================================================================
% input:rt_img_dir:  absolute root of data set
%       conv_idex:  index of used layers of CNN 
%       data_set_net:  name of data set
%       img_type: type of image (.tif,.jpg)
%       reduced_fc_num: number of reduced feature channels
%output:convfeacov: encoded feature(each column represents a image)

% wrritten by Nanjun He (henanjun@hnu.edu.cn)
%==========================================================================

disp('Enconding feature with stacked covariance pooling...');
subfolders = dir(rt_img_dir);
count = 1;
for ii = 1:length(subfolders)
    subname = subfolders(ii).name;
    if ~strcmp(subname, '.') & ~strcmp(subname, '..')
        frames = dir(fullfile(rt_img_dir, subname,img_type));
        c_num = length(frames);           
        for jj = 1:c_num
            imgpath = fullfile(rt_img_dir, subname, frames(jj).name);            
            I = imread(imgpath);
            I = single(I) ; % note: 255 range
            I = imresize(I, net.meta.normalization.imageSize(1:2),'bicubic') ;
            I = I -net.meta.normalization.averageImage ;
            % Run the CNN.
            yres = vl_simplenn(net, I) ;         
             if count==1
                for kk = 1:length(conv_idex)
                    sz(kk) = size(yres(conv_idex(kk)).x,1);
                end
             end
          [vv,idid] = sort(sz,'descend');
          resize_sz = [vv(end), vv(end)];% the height and width are the same
          stacked_covfea = [];
            for kk = 1:length(conv_idex)
                tmptmp = yres(conv_idex(kk)).x;
                tmptmp = imresize(tmptmp, resize_sz,'bilinear');
                tmptmp = average_fusion( tmptmp,reduced_fc_num );
                stacked_covfea = cat(3,stacked_covfea,tmptmp);
            end

            [m,n,d] = size(stacked_covfea);          
            tmp_mat = reshape(double(stacked_covfea),m*n,d)';
            [tmp_mat] = L2norm(tmp_mat);
            tmp_mat(isnan(tmp_mat)) = 0;
            mean_mat = mean(tmp_mat,2);
            centered_mat = tmp_mat-repmat(mean_mat,1,size(tmp_mat,2));
            tmp = centered_mat*centered_mat'/((size(tmp_mat,2))-1);
            tmp = tmp+0.005*eye(size(tmp)); % add small constant on the diognal entries to make sure the covariace matrix is strictly positive definite
            convfeacov(:,count) = SPD2Euclidean(logm(tmp));% logm operation and extact the enties in upper triangle 
            fprintf('Proc.%s-%s: wid_fm %d, hgt_fm%d, dpt_fm %d\n', data_set_name, frames(jj).name, m, n,d);
            count = count+1;
        end
        
    end
end
end 
