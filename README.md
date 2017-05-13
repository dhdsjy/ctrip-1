出行产品未来14个月销量预测大赛

团队：411的梦想少年
作者：郭安静

文件解释：
15808.py为19维特征的单模型XGB，其初赛得分为158.08分
15780.py为17维特征的单模型XGB，其初赛得分为157.80分
bagging_15746.py为15780模型的bagging,为10个模型在参数稍微抖动下的bagging，其初赛得分为157.46分，复赛得分为151.03分
Demo_last_15084.py为bagging_15746的结果和15808的结果的加权融合，复赛得分为150.84分。

使用方法：
１、运行15808.py，得到prediction_guoanjing_15808.txt，即158.08得分的预测结果
２、运行bagging_15746.py，得到prediction_guoanjing_bagging.txt，即157.46得分的预测结果
３、运行Demo_last_15084.py，得到prediction_guoanjing_15084.txt，即最终结果。

注意事项：
bagging的时候，因为每次参数抖动可能不一样，会导致结果出现细微差异，但复赛得分一般在151左右波动。
因为bagging是在比赛后期才做的，可能调试不够充分，如果出现较大差异，请测试15808和15780两个单模型的融合结果，步骤如下：
１、运行15808.py，得到prediction_guoanjing_15808.txt，即158.08得分的预测结果
２、运行15780.py，得到prediction_guoanjing_15780.txt，即157.80得分的预测结果
３、运行Demo_last_nobagging.py，得到prediction_guoanjing_nobagging.txt，即最终结果。

最佳结果：
最佳结果.txt即为得分150.84的提交文件

需要环境：
python:'2.7'
xgboost:'0.6'
numpy:'1.12.0'
pandas:'0.17.1'


额外特征参考网址：
月均气温、降水，天气网：http://www.tianqi.com/qiwen/china-spring/
特殊节假日，百度日历：https://www.baidu.com/baidu?wd=%E6%97%A5%E5%8E%86&tn=monline_dg&ie=utf-8

email:905303827@qq.com

