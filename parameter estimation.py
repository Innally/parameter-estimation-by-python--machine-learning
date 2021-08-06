# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

def kneighbor_est (test,X,k):
    topx = []
    info={}
    maxdis = 0
    for i in X:
        if i[0]==test[0] and i[1]==test[1]:
            continue
        dis=np.linalg.norm(test-i[0:2])
        if len(topx)<k:
            topx.append(dis)
            info[dis]=i[2]
            topx=sorted(topx)
            maxdis=topx[len(topx)-1]
        elif maxdis > dis:
            topx.append(dis)
            info[dis]=i[2]
            topx=sorted(topx)
            todel=topx.pop(k) # del the minimum
            info.pop(todel)
            maxdis=topx[k-1]

    lable=[0,0,0]
    for i in info.keys():
        if info[i]==1:
            lable[0]+=1
        elif info[i]==2:
            lable[1]+=1
        else:
            lable[2]+=1;

    return lable.index(max(lable))


def knn(mean, cov, P1, P2,k):
    error1=0
    error2=0
    X1 = Generate_DataSet(mean, cov, P1)
    X2 = Generate_DataSet(mean, cov, P2)
    for i in X1:
        lable=kneighbor_est(i[0:2],X1,k)
        if lable!=i[2]-1:
            error1+=1
    for i in X2:
        lable=kneighbor_est(i[0:2],X2,k)
        if lable!=i[2]-1:
            error2+=1
    return error1,error2

def Cal_gussWin(test,X,n,h): # 求的是和每一类中所有的点进行一个核函数的求值，然后哪个类最大，就选哪个类
    t=np.zeros(3)
    start=0
    for lable in range(1,4):
        for i in range(start,n):
            if X[i][0]==test[0] and X[i][1]==test[1]:
                continue
            if X[i][2] == lable:

                temp = test-X[i][0:2]
                norm = np.linalg.norm(temp)
                t[lable-1] += 1/np.sqrt(2*np.pi*h**2)*np.exp(-0.5*norm/h**2)
            else:
                start = i+1
                break

        lable += 1
    return t/n


def Core_func(X,class_num,h):
    num = np.array(X).shape[0] # 获得x中的点数量
    error_rate = 0
    for i in range(num):
        p_temp = Cal_gussWin(X[i][0:2],X,num,h)  # 计算样本i决策到j类的概率
        p_class = np.argmax(p_temp) + 1  # 得到样本i决策到的类
        if p_class != X[i][2]:
            error_rate += 1
    return error_rate/num



# 定义高斯函数，计算概率p(x|w)
def Gaussian_function(x, mean, cov):
    det_cov = np.linalg.det(cov)  # 计算方差矩阵的行列式
    inv_cov = np.linalg.inv(cov)  # 计算方差矩阵的逆
    # 计算概率p(x|w)
    p = 1 / (2 * np.pi * np.sqrt(det_cov)) * np.exp(-0.5 * np.dot(np.dot((x - mean), inv_cov), (x - mean)))
    return p


# 生成正态分布数据
def Generate_Sample_Gaussian(mean, cov, P, label):
    '''
        mean 为均值向量
        cov 为方差矩阵a
        P 为单个类的先验概率
        return 单个类的数据集
    '''
    temp_num = round(1000 * P)
    x, y = np.random.multivariate_normal(mean, cov, temp_num).T  # to generate a normal distribution by given condition
    z = np.ones(temp_num) * label
    X = np.array([x, y, z])
    return X.T


# 根据不同先验生成不同的数据集
def Generate_DataSet(mean, cov, P):
    # 按照先验概率生成正态分布数据
    # 返回所有类的数据集
    X = []
    label = 1
    for i in range(3):
        # 把此时类i对应的数据集加到已有的数据集中
        X.extend(Generate_Sample_Gaussian(mean[i], cov, P[i], label))
        label += 1
        i = i + 1
    return X


def Generate_DataSet_plot(mean, cov, P):
    # 画出不同先验对应的散点图
    xx = []
    label = 1
    for i in range(3):
        xx.append(Generate_Sample_Gaussian(mean[i], cov, P[i], label))
        label += 1
        i = i + 1
    # 画图
    plt.figure()
    for i in range(3):
        plt.plot(xx[i][:, 0], xx[i][:, 1], '.', markersize=4.)
        plt.plot(mean[i][0], mean[i][1], 'r*')
    return xx


# 似然率测试规则
def Likelihood_Test_Rule(X, mean, cov, P):
    class_num = mean.shape[0]  # 类的个数
    num = np.array(X).shape[0]
    error_rate = 0
    for i in range(num):
        p_temp = np.zeros(3)
        for j in range(class_num):
            p_temp[j] = Gaussian_function(X[i][0:2], mean[j], cov)   # 计算样本i决策到j类的概率
        p_class = np.argmax(p_temp) + 1  # 得到样本i决策到的类
        if p_class != X[i][2]:
            error_rate += 1
    return error_rate / num


##最大后验概率规则
def Max_Posterior_Rule(X, mean, cov, P):
    class_num = mean.shape[0]  # 类的个数
    num = np.array(X).shape[0]
    error_rate = 0
    for i in range(num):
        p_temp = np.zeros(3)
        for j in range(class_num):
            p_temp[j] = Gaussian_function(X[i][0:2], mean[j], cov) * P[j]  # 计算样本i是j类的后验概率
        p_class = np.argmax(p_temp) + 1  # 得到样本i分到的类
        if p_class != X[i][2]:
            error_rate += 1
    return error_rate / num


# 单次试验求不同准则下的分类误差
def repeated_trials(mean, cov, P1, P2):
    # 根据mean，cov，P1,P2生成数据集X1,X2
    # 通过不同规则得到不同分类错误率并返回
    # 生成N=1000的数据集
    X1 = Generate_DataSet(mean, cov, P1)
    X2 = Generate_DataSet(mean, cov, P2)
    error = np.zeros((2, 2))
    # 计算似然率测试规则误差
    error_likelihood = Likelihood_Test_Rule(X1, mean, cov, P1)
    error_likelihood_2 = Likelihood_Test_Rule(X2, mean, cov, P2)
    error[0] = [error_likelihood, error_likelihood_2]
    # 计算最大后验概率规则误差
    error_Max_Posterior_Rule = Max_Posterior_Rule(X1, mean, cov, P1)
    error_Max_Posterior_Rule_2 = Max_Posterior_Rule(X2, mean, cov, P2)
    error[1] = [error_Max_Posterior_Rule, error_Max_Posterior_Rule_2]

    core_error1=[]
    core_error2=[]
    for h in [0.5,1,1.5,2]:
        core_error1.append(Core_func(X1,mean.shape[0],h))
        core_error2.append(Core_func(X2,mean.shape[0],h))
    core_error1=np.array(core_error1)
    core_error2=np.array(core_error2)

    return error,core_error1,core_error2


if __name__ == '__main__':
    mean = np.array([[1, 1], [4, 4], [8, 1]])  # 均值数组
    cov = [[2, 0], [0, 2]]  # 方差矩阵
    num = 1000  # 样本个数
    P1 = [1 / 3, 1 / 3, 1 / 3]  # 样本X1的先验概率
    P2 = [0.6, 0.3, 0.1]  # 样本X2的先验概率
    Generate_DataSet_plot(mean, cov, P1)  # 画X1数据集散点图
    Generate_DataSet_plot(mean, cov, P2)  # 画X2数据散点图
    # 计算十次运算的总误差
    error_all = np.zeros((2, 2))
    core_all1=np.zeros((1,4))
    core_all2=np.zeros((1,4))
    # 测试times_num次求平均
    times_num = 10

    for times in range(times_num):
        terro,core1,core2=repeated_trials(mean,cov,P1,P2)
        print("round",times)
        print("terro",terro)
        print("core1",core1)
        print("core2",core2)
        error_all += terro
        core_all1 += core1
        core_all2 += core2

    # 计算平均误差
    ave_error1 = error_all / times_num
    ave_core_error1 = core_all1 / times_num
    ave_core_error2 = core_all2 / times_num
    print("average error 1 is", ave_error1)
    print("average core error 1 is", ave_core_error1)
    print("average core error 2 is", ave_core_error2)
    ks=[1,3,5]
    for k in ks:
        err1,err2=knn(mean,cov,P1,P2,k)
        print("---------k=",k)
        print("erro1:",err1/1000.0)
        print("erro2:",err2/1000.0)