data = heartdiseasedata; 
feat = data(:,1:13); % feature matrix
label = data(:,14);  % class label vector
C = unique(label); %extract label information from label vector
 
%-----2. Prepare N-fold dataset for classification----------%
N = 5; % N-fold cross validation 
data_nfold = divide_nfold_data(feat, label, N); 
 
%-----3. Perform N-fold Cross-Validation using KNN Function-----------------% 
ACC_SUM = [];
 
 
for K = [3 5 7] 
    for Dorder = [1 2 5] 
        acc_nfold = [];
        Lpred_nfold =[];
        Ltest_nfold =[];
        confusion_nfold = zeros(3,3);
        for ifold = 1:N 
           %----prepare cross-validation training and testing dataset---% 
           idx_test = ifold; % index for testing fold
           idx_train = setdiff(1:N, ifold); % index for training folds
           Dtest = []; Ltest = []; % initialize testing data and label
           Dtrain = []; Ltrain = []; % initialize testing data and label
 
           %---construct the training and testing dataset for the ith fold cross validatoin
           for iC = 1:length(C) 
               cl = C(iC);   
               dtest = eval(['data_nfold.class',num2str(cl), '.fold', num2str(ifold)]);
               Dtest = [Dtest; dtest]; 
               Ltest = [Ltest; cl*ones(size(dtest,1), 1)]; 
 
               for itr = 1:length(idx_train)
                   idx = idx_train(itr); 
                   dtrain = eval(['data_nfold.class',num2str(cl), '.fold', num2str(idx)]);
                   Dtrain = [Dtrain; dtrain];
                   Ltrain = [Ltrain; cl*ones(size(dtrain,1), 1)]; 
               end  
           end
           %% change opt here for LDA=3 &QDA=2%%
       opt = 1; 
       Lpred = myBayesPredict(Dtrain, Ltrain, Dtest, opt); 
           %---Calculate Classification Accuracy-----%
           acc = sum(Lpred==Ltest)/length(Ltest);
           if ifold ==1
               Lpred1 =Lpred;
               Ltest1 =Ltest;
           elseif ifold ==2
               Lpred2 =Lpred;
               Ltest2 =Ltest;
           elseif ifold ==3
               Lpred3 =Lpred;
               Ltest3 =Ltest;
           elseif ifold ==4
               Lpred4 =Lpred;
               Ltest4 =Ltest;
           elseif ifold ==5
               Lpred5 =Lpred;
               Ltest5 =Ltest;
           end
           acc_nfold(ifold, 1) = acc; 
	 
        end
 	 acc_ave = mean(acc_nfold); 
        ACC_SUM = [ACC_SUM; K, Dorder, acc_ave];
        confusionmat(Ltest,Lpred);
        confusionchart(Ltest,Lpred);
    end
end
