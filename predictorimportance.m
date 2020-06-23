t = templateTree('MaxNumSplits',1,'Surrogate','on');
ens = fitrensemble(heartdiseasedata(:,1:13),heartdiseasedata(:,14),'Method','LSBoost','Learners',t);
[imp,ma] = predictorImportance(ens);
imp = predictorImportance
