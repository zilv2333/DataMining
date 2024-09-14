import pandas as pd
import numpy as np
from func.calculate import *

class Kmeans:
    """

    """
    def __init__(self,k:int,metric=EuclideanDistance):
        self.k=k
        self.data=None
        self.center=None
        self.metric=metric

    def fit(self,data,epochs,inplace=False):
        if inplace:
            self.data=data
        else:
            self.data=pd.DataFrame(data)
        if self.center:
            chosen_data=self.center
        else:
            chosen_data=data.sample(self.k)
        distance=pd.DataFrame()
        for epoch in range(epochs):
            for i in range(chosen_data.shape[0]):
                distance["distance{}".format(i)]=self.metric(chosen_data.iloc[i,:],self.data.iloc[:,:chosen_data.shape[1]])
            self.data["class"]=distance.idxmin(axis=1).apply(lambda x:x[-1]).astype(int)
            for i in range(chosen_data.shape[0]):
                chosen_data.iloc[i,:]=cal_center(self.data.iloc[:,0:4].where(self.data["class"]==i).dropna().values)
                # print(chosen_data)

        self.center=chosen_data.reset_index(drop=True)


        return self.data

    def class_center(self):
        return self.center

    def predict(self,test):
        model=Knn(self.k,self.metric)
        model.fit(self.data)
        return model.predict(test)






class Knn:

    """
  利用knn算法得到测试点所属类别
  :param Test: 测试点
  :param Data: dataframe类型，规定最后一列为类别
  :param k: k的取值
  :return: 类别
  """
    def __init__(self,k:int,metric=EuclideanDistance):
        self.k=k
        self.data=None
        self.metric=metric

    def fit(self,data):
        self.data=data

    def predict(self,Test):

        Test=np.array(Test)
        distance=pd.DataFrame(self.data.iloc[:, -1].values, columns=[ "class"])
        for i in range(Test.shape[0]):

            distance=pd.concat([distance,pd.DataFrame(self.metric(Test[i],self.data.iloc[:,:-1]),columns=["distance{}".format(i)])],axis=1)
            # distance["distance{}".format(i)]=self.metric(Test[i],self.data.iloc[:,:-1])
            # print(distance)
        res=[]
        # print(distance)

        for i in range(Test.shape[0]):
            distance_d=distance.sort_values(by=["distance{}".format(i)]).iloc[0:self.k].reset_index(drop=True)
            classCount={}
            # print(distance_d)
            for i in range(self.k):
                classCount[distance_d["class"][i]]=classCount.get(distance_d["class"][i], 0)+1
            sortedClassCount = sorted(classCount.items(),key=lambda x:x[1],reverse=True)
            res.append(sortedClassCount[0][0])
        return res


# model=Knn(k=3)


