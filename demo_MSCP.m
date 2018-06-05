clc;clear;
dataset_name =  'UCM21'; % UCM21; AID30;WHU19;NWPU45;
img_type =  '*.tif';% UCM21='.tif', AID30='.jpg';WHU19='.jpg';NWPU45='.jpg';
net_name = 'imagenet-vgg-verydeep-16'; %imagenet-vgg-verydeep-16; imagenet-caffe-alex；
rt_img_dir = ['D:\matlab_work_folder\classification\Scene_classification\Scene_data\',dataset_name,'\'];


switch(net_name)
    case 'imagenet-caffe-alex'
        conv_idex = [10,12,14];
        fc_num = 80;
    case 'imagenet-vgg-verydeep-16'
        conv_idex = [16 23 30];
        fc_num = 130;
end

switch(dataset_name)
    case 'UCM21'
        no_classes = 21;
        allsample_perclass = ones(1,no_classes)*100;
    case 'AID30'
        no_classes = 30;
        allsample_perclass = [360,310,220,400,360,260,240,350,410,300,...
                                               370,250,390,280,290,340,350,390,370,420,...
                                               380,260,290,410,300,300,330,290,360,420] ;
    case 'NWPU45'
        no_classes = 45;
        allsample_perclass = ones(1,no_classes)*700;       
end

net = load(net_name) ;
net = vl_simplenn_tidy(net) ;

%% stage one: feature enconding with stacked convariance pooling

[convfeacov] = calc_stackedCovdescriptor(rt_img_dir,net,conv_idex, dataset_name, img_type, fc_num-1);


labels = [];
for i = 1:no_classes
    labels = [labels,ones(1,allsample_perclass(i))*i];
end

numTesteach = 500;
train_number = ceil(allsample_perclass*0.8);

for i = 1:10
    [train{i},test{i},test_number]= GenerateSample(labels',train_number,no_classes);
    train_SL = train{i};
    test_SL = test{i};
    train_id = train_SL(1,:);
    train_label = train_SL(2,:);
    test_id = test_SL(1,:);
    test_label = test_SL(2,:);
    param.kernel = 'Log-E inner';
    if length(test_id)<=numTesteach
        train_cov = convfeacov(:,train_id);
        test_cov = convfeacov(:,test_id);
        [ KMatrix_Train ] = double(makekernl(train_cov', train_cov', param));
        [ KMatrix_Test ] = double(makekernl(train_cov', test_cov', param));
        Ktrain = [(1:size(KMatrix_Train,1))',KMatrix_Train];     %样本的序列号放在核矩阵前面  
        model = svmtrain(train_label', Ktrain, '-t 4 -b 1');  % 输入 Ktrain    %求测试集核矩阵  
        Ktest = [(1:size(KMatrix_Test,2))', KMatrix_Test']; %样本的序列号放在核矩阵前面  
        tmp = ones(1,size(test_cov,2));
        [predict_label, accuracy, P1] = svmpredict(tmp',Ktest,model,'-b 1'); % 输入Ktest
        [OA(i),Kappa,AA,CA(:,i),cfm(:,:,i)] = calcError(test_SL(2,:)'-1,predict_label-1,[1:no_classes]);
    else
        train_cov = convfeacov(:,train_id);
        predict_label = [];
        [KMatrix_Train ] = double(makekernl(train_cov', train_cov', param));
        numsplit = ceil(length(test_id)/numTesteach);
        for nsplit = 1:numsplit
            if nsplit == numsplit
                tmp_id = test_id((nsplit-1)*numTesteach+1:end);
            else
                tmp_id = test_id((nsplit-1)*numTesteach+1:nsplit*numTesteach);
            end
            test_cov = convfeacov(:,tmp_id);
            [ KMatrix_Test ] = double(makekernl(train_cov', test_cov', param));
            Ktrain = [(1:size(KMatrix_Train,1))',KMatrix_Train];     %样本的序列号放在核矩阵前面  
            model = svmtrain(train_label', Ktrain, '-t 4 -b 1');  % 输入 Ktrain    %求测试集核矩阵  
            Ktest = [(1:size(KMatrix_Test,2))', KMatrix_Test']; %样本的序列号放在核矩阵前面  
            tmp = ones(1,size(test_cov,2));
            [predict_label_tmp, accuracy, P1] = svmpredict(tmp',Ktest,model,'-b 1'); % 输入Ktest
            predict_label = [predict_label;predict_label_tmp];
        end
        [OA(i),Kappa,AA,CA(:,i), cfm(:,:,i)] = calcError(test_SL(2,:)'-1,predict_label-1,[1:no_classes]);
    end
end
jilu(1,1) = mean(OA);
jilu(1,2) = std(OA);
jilu(2:no_classes+1,1) = mean(CA,2);
jilu(2:no_classes+1,2) = std(CA,[],2);


