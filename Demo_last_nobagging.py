#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:44:42 2017

@author: gaj
"""
import pandas as pd
############15780和15808的融合结果###########################################

a=pd.read_csv('./prediction_guoanjing_15780.txt')['ciiquantity_month']
c=pd.read_csv('./prediction_guoanjing_15808.txt')['ciiquantity_month']
d=a*0.8+c*0.2
answer_table=pd.read_csv('prediction_guoanjing_201703251.txt')
answer_table.ciiquantity_month=d
answer_table.to_csv('prediction_guoanjing_nobagging.txt',index=False)