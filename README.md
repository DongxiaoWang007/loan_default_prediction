### 本案例针对kaggle上提供的某金融机构的历史贷款数据, 建立相应的模型用于预测贷款违约的行为, 期望在发放贷款前能够对贷款违约的风险作出衡量和预测, 更加明晰地明确影响贷款违约风险的特征.

以下内容为该案例的代码运行的结果:  
![result1.png]('/images/result1.png')  
![result2.png]('/images/result2.png')  


TRAIN: [ 17941 117875  67893 ...  93992  20627   5744]   
TEST: [ 64422 113530  30105 ...  34862 130492 127209]

auc= 0.8050297368318645  
auc= 0.8121010624209538  
auc= 0.9999992216110204  
auc= 0.8121010624209538  

[(0.3266, 'RUUnsecuredL'), (0.1698, 'NOTimes90'), (0.1674, 'NOTime30-59'), (0.0783, 'NOTime60-89'), (0.0699, 'age'), (0.0644, 'DebtRatio'), (0.0486, 'Income'), (0.0428, 'NOCredit'), (0.0207, 'NORealEstate'), (0.0115, 'NODependents')]

01) RUUnsecuredL                   0.326557  
02) age                            0.169813  
03) NOTime30-59                    0.167416  
04) DebtRatio                      0.078330  
05) Income                         0.069923  
06) NOCredit                       0.064409  
07) NOTimes90                      0.048556  
08) NORealEstate                   0.042807  
09) NOTime60-89                    0.020651  
10) NODependents                   0.011539  

the best parameter:  
RandomForestClassifier(bootstrap=True, class_weight='balanced_subsample',
                       criterion='gini', max_depth=None, max_features=2,
                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=50,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=100, n_jobs=-1, oob_score=True,
                       random_state=None, verbose=0, warm_start=False)  
                       
the best score: 0.8631859891147633  

auc= 0.9072155181794597  
auc= 0.8644198099216118