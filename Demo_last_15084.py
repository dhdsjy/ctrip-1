# -*- coding: utf-8 -*-
"""
Created on Sun Apr  2 08:48:52 2017

@author: gaj
"""

import pandas as pd

a=pd.read_csv('./prediction_guoanjing_bagging.txt')['ciiquantity_month']
c=pd.read_csv('./prediction_guoanjing_15808.txt')['ciiquantity_month']
d=a*0.8+c*0.2
answer_table=pd.read_csv('prediction_guoanjing_201703251.txt')
answer_table.ciiquantity_month=d
answer_table.to_csv('prediction_guoanjing_15084.txt',index=False)
