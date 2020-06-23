function Lpred = myBayesPredict(Dtrain, Ltrain, Dtest, opt)


%% 1. Use Naive Bayes Function to Make classification 
% Assume the features are independent, then we can use Naive Bayes for prediction
if opt==1
NB = fitcnb(Dtrain,Ltrain);  % construct a Naive Bayes model NB
Lpred = predict(NB, Dtest);  % apply the trained model NB to predict class of test samples in Dtest
end


%% 2. QDA

if opt==2
   C = unique(Ltrain); 
   Lpred = []; 
   
   for iC = 1:length(C) % For each class i, calculate P(X|Wj)P(Wj) for all testing samples
       cl = C(iC);  
       idx = find(Ltrain==cl); 
       data = Dtrain(idx,:); 
       mu = mean(data); % feature mean vector
       sigma = cov(data); % feature covariance matrix
       P = length(idx)/length(Ltrain);
       
       % For each testing sample, calculate P(X|Wj)P(Wj) = likelihood of class i * prior of class i
       for j = 1:size(Dtest,1) 
           x = Dtest(j, :); 
           likelihood = mvnpdf(x,mu,sigma); % likelihood of the current class i
           prior = P; % prior of the current class i
           G(iC, j) = likelihood*prior; % P(X|Wj)P(Wj) 
       end
   end
   
   % For each testing sample, find the index of the class that have maximum
   % value of likelihood*prior
   [~, pred] = max(G);   
   Lpred = C(pred);    
end 



%% 3. LDA
if opt==3
   C = unique(Ltrain); 
   Lpred = [];
   
   for iC = 1:length(C) 
       cl = C(iC);  
       idx = find(Ltrain==cl); 
       data = Dtrain(idx,:); 
       
       mu = mean(data)'; 
       E = cov(data);
       P = length(idx)/length(Ltrain); 
       W = -0.5*inv(E); 
       w = inv(E)*mu; 
       w0 = -0.5*mu'*inv(E)*mu-0.5*log(det(E))+log(P); 
       
       for j = 1:size(Dtest,1)
           x = Dtest(j, :)'; 
           G(iC, j) = x'*W*x + w'*x + w0; 
       end
   end 
   
   [~, pred] = max(G); 
   Lpred = C(pred); 

end


 
