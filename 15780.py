# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 08:48:52 2017

@author: gaj
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import operator
import xgboost as xgb

def ceate_feature_map(features):  
    outfile = open('xgb.fmap', 'w')  
    i = 0  
    for feat in features:  
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))  
        i = i + 1  
    outfile.close()  

def regularization(table,name):
#    mean, std = table[name].mean(), table[name].std()
#    table.loc[:, name] = (table[name] - mean)/std

#    max_v=table[name].max()
#    min_v=table[name].min()
#    table.loc[:,name]=(table[name]-min_v)/(max_v-min_v)
    return table
    
def one_hot(table,name):
    dummies = pd.get_dummies(table[name], prefix=name)
    table = pd.concat([table, dummies], axis=1)
    table = table.drop(name, axis=1)
    return table
    
def Pretreatment(table):#collect product_month
    table['month']=table['product_month'].apply(lambda x:x[5:7])
    
#    table=one_hot(table,'month')
    
    table['month']=table['month'].apply(lambda x:float(x))
    table=regularization(table,'month')
    
    
    table=table.drop('product_month', axis=1)    
    return table
      
def get_data(table,targets):    
    _features = table.drop(targets,axis=1).as_matrix()
    _targets = table[targets].as_matrix()
    _targets.shape = (_targets.shape[0], 1)
    _targets.transpose()
    return _features,_targets

#def get_data(table,targets):    
#    _features = table.drop(targets,axis=1).as_matrix()
#    _targets = table[targets].as_matrix()
#    _targets.shape = (_targets.shape[0], 1)
#    max_targets=np.max(_targets)
#    _targets=np.float32(_targets)/max_targets
#    return _features,_targets     
      
####################################main##############################################
evaluation=pd.read_csv('prediction_guoanjing_201703251.txt',encoding="utf-8-sig")
product_info=pd.read_csv('product_info.txt',encoding="utf-8-sig")
product_quantity=pd.read_csv('product_quantity.txt',encoding="utf-8-sig")


#holiday={'2014-01':8,'2014-02':11,'2014-03':10,'2014-04':9,'2014-05':10,'2014-06':10,\
#        '2014-07':8,'2014-08':10,'2014-09':8,'2014-10':12,'2014-11':10,'2014-12':8,\
#        '2015-01':9,'2015-02':11,'2015-03':8,'2015-04':9,'2015-05':11,'2015-06':9,\
#        '2015-07':8,'2015-08':10,'2015-09':9,'2015-10':13,'2015-11':9,'2015-12':8,\
#        '2016-01':11,'2016-02':11,'2016-03':8,'2016-04':10,'2016-05':10,'2016-06':9,\
#        '2016-07':10,'2016-08':8,'2016-09':8,'2016-10':13,'2016-11':8,'2016-12':9,\
#        '2017-01':12}

#just special holiday  
special_holiday={'2014-01':2,'2014-02':6,'2014-03':0,'2014-04':3,'2014-05':4,'2014-06':2,\
        '2014-07':0,'2014-08':0,'2014-09':3,'2014-10':7,'2014-11':0,'2014-12':0,\
        '2015-01':3,'2015-02':7,'2015-03':0,'2015-04':3,'2015-05':3,'2015-06':3,\
        '2015-07':0,'2015-08':0,'2015-09':5,'2015-10':7,'2015-11':0,'2015-12':0,\
        '2016-01':3,'2016-02':7,'2016-03':0,'2016-04':3,'2016-05':2,'2016-06':3,\
        '2016-07':0,'2016-08':0,'2016-09':3,'2016-10':7,'2016-11':0,'2016-12':1,\
        '2017-01':7}

students_holiday={'2014-01':10,'2014-02':30,'2014-03':0,'2014-04':0,'2014-05':0,'2014-06':0,\
        '2014-07':30,'2014-08':30,'2014-09':0,'2014-10':0,'2014-11':0,'2014-12':0,\
        '2015-01':10,'2015-02':30,'2015-03':0,'2015-04':0,'2015-05':0,'2015-06':0,\
        '2015-07':30,'2015-08':30,'2015-09':0,'2015-10':0,'2015-11':0,'2015-12':0,\
        '2016-01':10,'2016-02':30,'2016-03':0,'2016-04':0,'2016-05':0,'2016-06':0,\
        '2016-07':30,'2016-08':30,'2016-09':0,'2016-10':0,'2016-11':0,'2016-12':0,\
        '2017-01':10}

#china temperature one month
month_temperature={'2014-01':0,'2014-02':3,'2014-03':8,'2014-04':15,'2014-05':16,'2014-06':23,\
        '2014-07':26,'2014-08':25,'2014-09':21,'2014-10':15,'2014-11':8,'2014-12':3,\
        '2015-01':0,'2015-02':3,'2015-03':8,'2015-04':15,'2015-05':16,'2015-06':23,\
        '2015-07':26,'2015-08':25,'2015-09':21,'2015-10':15,'2015-11':8,'2015-12':3,\
        '2016-01':0,'2016-02':3,'2016-03':8,'2016-04':15,'2016-05':16,'2016-06':23,\
        '2016-07':26,'2016-08':25,'2016-09':21,'2016-10':15,'2016-11':8,'2016-12':3,\
        '2017-01':0
        }


#month_season={'2014-01':4,'2014-02':1,'2014-03':1,'2014-04':1,'2014-05':2,'2014-06':2,\
#        '2014-07':2,'2014-08':3,'2014-09':3,'2014-10':3,'2014-11':4,'2014-12':4,\
#        '2015-01':4,'2015-02':1,'2015-03':1,'2015-04':1,'2015-05':2,'2015-06':2,\
#        '2015-07':2,'2015-08':3,'2015-09':3,'2015-10':3,'2015-11':4,'2015-12':4,\
#        '2016-01':4,'2016-02':1,'2016-03':1,'2016-04':1,'2016-05':2,'2016-06':2,\
#        '2016-07':2,'2016-08':3,'2016-09':3,'2016-10':3,'2016-11':4,'2016-12':4,\
#        '2017-01':4
#        }

#month_rain
month_rain={'2014-01':21,'2014-02':29,'2014-03':46,'2014-04':70,'2014-05':101,'2014-06':134,\
        '2014-07':156,'2014-08':140,'2014-09':85,'2014-10':49,'2014-11':29,'2014-12':17,\
        '2015-01':21,'2015-02':29,'2015-03':46,'2015-04':70,'2015-05':101,'2015-06':134,\
        '2015-07':156,'2015-08':140,'2015-09':85,'2015-10':49,'2015-11':29,'2015-12':17,\
        '2016-01':21,'2016-02':29,'2016-03':46,'2016-04':70,'2016-05':101,'2016-06':134,\
        '2016-07':156,'2016-08':140,'2016-09':85,'2016-10':49,'2016-11':29,'2016-12':17,\
        '2017-01':21}


product_quantity['product_month']=product_quantity['product_date'].apply(lambda x: x[:7])
evaluation['product_month']=evaluation['product_month'].apply(lambda x: x[:7])

train_month=product_quantity.groupby(['product_id','product_month']).sum()['ciiquantity']
 
train_month=pd.DataFrame(train_month)
train_month=train_month.reset_index()

#holiday
#train_month['holiday']=train_month['product_month'].apply(lambda x:holiday[x])
#evaluation['holiday']=evaluation['product_month'].apply(lambda x:holiday[x])

#special_holiday
train_month['special_holiday']=train_month['product_month'].apply(lambda x:special_holiday[x])
evaluation['special_holiday']=evaluation['product_month'].apply(lambda x:special_holiday[x])
#
train_month['students_holiday']=train_month['product_month'].apply(lambda x:students_holiday[x])
evaluation['students_holiday']=evaluation['product_month'].apply(lambda x:students_holiday[x])
#
train_month['month_temperature']=train_month['product_month'].apply(lambda x:month_temperature[x])
evaluation['month_temperature']=evaluation['product_month'].apply(lambda x:month_temperature[x])
#
train_month['month_rain']=train_month['product_month'].apply(lambda x:month_rain[x])
evaluation['month_rain']=evaluation['product_month'].apply(lambda x:month_rain[x])

product_info_use=product_info.drop([ 'eval4','district_id3','district_id4','railway'
                                    , 'airport', 'citycenter', 'railway2', 'airport2','citycenter2'
                                    , 'upgradedate'],axis=1)

#for x in ['eval', 'eval2', 'eval3','voters','maxstock','lon','lat']:
#    product_info_use=regularization(product_info_use,x)
    
#for x in ['district_id1','district_id2']:
#    product_info_use=one_hot(product_info_use,x)
    
#for x in ['holiday','special_holiday']:
#    train_month=regularization(train_month,x) 
#    evaluation=regularization(evaluation,x)
    
#######################################合作时间####################################
#coop=product_info_use['cooperatedate']
#n_month_coop=np.zeros((coop.shape[0],1),dtype=np.float32)
#for i in xrange(coop.shape[0]):
#    temp=coop[i]
#    if temp==u'1753-01-01':
#        n_month_coop[i,0]=-1
#    elif temp==u'-1':
#        n_month_coop[i,0]=-1
#    else:
#        year=int(coop[i][0:4])
#        month=int(coop[i][5:7])
#        n_month_coop[i,0]=(2017-year)*12+12-month
#
#product_info_use['coopmonth']=n_month_coop
product_info_use=product_info_use.drop(['cooperatedate'],axis=1)

##################################测试产品开始日期#################################

#start=product_info_use['startdate']
#n_month_start=np.zeros((start.shape[0],1),dtype=np.float32)
#for i in xrange(start.shape[0]):
#    temp=start[i]
#    if temp==u'1753-01-01':
#        n_month_start[i,0]=-1
#    elif temp==u'-1':
#        n_month_start[i,0]=-1
#    else:
#        year=int(start[i][0:4])
#        month=int(start[i][5:7])
#        n_month_start[i,0]=(2017-year)*12+12-month
#
#product_info_use['startmonth']=n_month_start
product_info_use=product_info_use.drop(['startdate'],axis=1)

##########################获取基础月##################################
train=train_month.as_matrix()
n_month=np.zeros((train.shape[0],1))

for i in range(train.shape[0]):
    base_year=2014
    base_month=1
    
    this_year=int(train[i,1][0:4])
    this_month=int(train[i,1][5:7])
    
    n_month[i,0]=(this_year-base_year)*12+(this_month-base_month)+1

train_month['base_month']=n_month

############################测试基础月#####################################
evl=evaluation.as_matrix()
n_month=np.zeros((evl.shape[0],1))

for i in range(evl.shape[0]):
    base_year=2014
    base_month=1
    
    this_year=int(evl[i,1][0:4])
    this_month=int(evl[i,1][5:7])
    
    n_month[i,0]=(this_year-base_year)*12+(this_month-base_month)+1

evaluation['base_month']=n_month


train_month=Pretreatment(train_month)
evaluation=Pretreatment(evaluation) 

train_month=pd.merge(train_month,product_info_use,on='product_id',how='left')
evaluation=pd.merge(evaluation,product_info_use,on='product_id',how='left')


train_month=regularization(train_month,'product_id')
evaluation=regularization(evaluation,'product_id')

#cal number of nan in feature vector
n_nan=train_month[train_month.iloc[:,:]==-1].sum(axis=1)
n_nan=pd.DataFrame(n_nan)
n_nan=n_nan.fillna(value=0)
#
train_month['nna']=n_nan
evaluation['nna']=n_nan

#train_month=train_month.drop('product_id',axis=1)
#evaluation=evaluation.drop('product_id',axis=1)
#print(train_month.shape,evaluation.shape)

train_month_x,train_month_y = get_data(train_month,'ciiquantity')
evaluation_x,evaluation_y = get_data(evaluation,'ciiquantity_month')

#mixind=range(train_month_x.shape[0])
#random.shuffle(mixind)
#train_month_x=train_month_x[mixind,:]
#train_month_y=train_month_y[mixind,:]

print(train_month_x.shape,train_month_y.shape)
print(evaluation_x.shape,evaluation_y.shape)

xgb_val = xgb.DMatrix(train_month_x[:,:],label=train_month_y[:])
xgb_train = xgb.DMatrix(train_month_x[0:,:], label=train_month_y[0:])
xgb_test = xgb.DMatrix(evaluation_x)


params={
'booster':'gbtree',
'objective': 'reg:linear', 
'gamma':0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
'max_depth':10, # 构建树的深度，越大越容易过拟合
'lambda':2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
'subsample':0.7, # 随机采样训练样本
'colsample_bytree':0.7, # 生成树时进行的列采样
'min_child_weight':3, 
# 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
#，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
#这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。 
'silent':0 ,#设置成1则没有运行信息输出，最好是设置为0.
'eta': 0.007, # 如同学习率
'seed':1000,
'nthread':4,# cpu 线程数
#'eval_metric': 'auc'
}
plst = list(params.items())
num_rounds = 1800 # 迭代次数
watchlist = [(xgb_train, 'train'),(xgb_val, 'val')]

#训练模型并保存
# early_stopping_rounds 当设置的迭代次数较大时，early_stopping_rounds 可在一定的迭代次数内准确率没有提升就停止训练
model = xgb.train(plst, xgb_train, num_rounds, watchlist,early_stopping_rounds=100)
#model.save_model('./model_xgboost/xgb.model') # 用于存储训练出的模型
print "best best_ntree_limit",model.best_ntree_limit 

answer_table=pd.read_csv('prediction_guoanjing_201703251.txt')
preds = model.predict(xgb_test,ntree_limit=model.best_ntree_limit)
answer_table.ciiquantity_month=preds
answer_table.to_csv('prediction_guoanjing_15780.txt',index=False)

features = [x for x in train_month.columns if x not in ['ciiquantity']]  
ceate_feature_map(features)  
  
importance = model.get_fscore(fmap='xgb.fmap')  
importance = sorted(importance.items(), key=operator.itemgetter(1))  
  
df = pd.DataFrame(importance, columns=['feature', 'fscore'])  
df['fscore'] = df['fscore'] / df['fscore'].sum()  
df.to_csv("./feat_importance.csv", index=False)  
  
plt.figure()  
df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 5))  
plt.title('XGBoost Feature Importance')  
plt.xlabel('relative importance')  
plt.show()

