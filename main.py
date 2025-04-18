import time
import warnings

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

import GBG_plusplus


#Determine the label of the test sample
def GB_KNN(X_test, Ball_list):
    predict_label = []
    ball_centers = []
    ball_num_list = []
    for ball in Ball_list:
        ball_centers.append(ball.center)
        ball_num_list.append(ball.num)
    samp_all = np.sum(ball_num_list)
    ball_density_list = ball_num_list / samp_all
    for row in X_test:
        dis = (GBG_plusplus.calculateDist(row, ball_centers) - ball_density_list).tolist()
        predict_ball = Ball_list[dis.index(min(dis))]
        predict_label.append(predict_ball.label)
    return predict_label, samp_all


# Visualize GBs (with 2 features)
def plot_gb_2(granular_ball_list):
    color = {1: 'k', -1: 'r'}
    plt.figure(figsize=(5, 4))
    plt.axis([0, 1, 0, 1])
    for granular_ball in granular_ball_list:
        label = granular_ball.label
        center, radius = granular_ball.center, granular_ball.radius
        data0 = granular_ball.data[granular_ball.data[:, 0] == 1]
        data1 = granular_ball.data[granular_ball.data[:, 0] == -1]
        plt.plot(data0[:, 1], data0[:, 2], '.', color=color[1], markersize=5)
        plt.plot(data1[:, 1], data1[:, 2], '.', color=color[-1], markersize=5)
        theta = np.arange(0, 2 * np.pi, 0.01)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        plt.plot(x, y, color[label], linewidth=0.8)
        plt.plot(center[0], center[1], 'x', color=color[label])

    plt.show()


def main(purity=1.0):
    warnings.filterwarnings("ignore")  # Ignore warnings
    data_list = ['parkinsons']
    n_splits = 10  #Number of folds crossed by N folds
    #Noise_ratio = 0  #The Noise Ratio of Class Label Noise Data
    file = open(str(purity) + 'GBKNN_plusplus' + '.txt', mode='a')
    #file.write('Noise_ratio: ' + str(Noise_ratio) + '\n')
    for data_nm in data_list:
        file.write(data_nm + '\n')
        data_frame = pd.read_csv(r"./" + data_nm + ".csv",header=None)  # 加载数据集
        data = data_frame.values  
        data_temp = []
        data[data[:, 0] == -1, 0] = 0
        data_list = data.tolist()
        data = []
        for data_single in data_list:
            if data_single[1:] not in data_temp:
                data_temp.append(data_single[1:])
                data.append(data_single)
        data = np.array(data)
        numberSample = data.shape[0]
        minMax = MinMaxScaler()
        data = np.hstack((data[:, 0].reshape(numberSample, 1),
                          minMax.fit_transform(data[:, 1:])))  
        train_data = data[:, 1:]  
        train_target = data[:, 0]  
        skf = StratifiedKFold(n_splits, shuffle=True,
                              random_state=42)  
        sum_acc, sum_f1, sum_ball,  sum_samp = 0.0, 0.0, 0, 0
        times = 0.0
        for train_index, test_index in skf.split(train_data, train_target):
            train, test = data[train_index], data[test_index]
            X_test = test[:, 1:]
            Y_test = test[:, 0]
            start = time.time()
            Ball_list = GBG_plusplus.generateGBList(train, purity)
            pre_test, samp_num = GB_KNN(X_test, Ball_list)
            end = time.time()
            sum_samp += samp_num
            ball_num = len(Ball_list)
            acc = accuracy_score(Y_test, pre_test)
            f1 = f1_score(Y_test, pre_test, average='macro')
            sum_f1 += f1
            sum_acc += acc
            times += end - start
            sum_ball += ball_num
            file.write(('accuracy of ' + data_nm + ':' + str(acc) + '\n'))
            file.write(('f1 of ' + data_nm + ':' + str(f1) + '\n'))
        file.write(('Average accuracy of ' + data_nm + ':' +
                    str(sum_acc / n_splits) + '\n'))
        file.write(
            ('Average f1 of ' + data_nm + ':' + str(sum_f1 / n_splits) + '\n'))
        file.write(('Average time of ' + data_nm + ':' +
                    str(times * 1000 / n_splits) + '\n'))
        file.write(('Average number of balls for ' + data_nm + ':' +
                    str(sum_ball / n_splits) + '\n'))
        file.write(('Average sample keep rate of ' + data_nm + ':' +
                    str(sum_samp / (len(train) * n_splits)) + '\n'))
        print(data_nm, ' done!')
    file.write('all done!!!!!')
    file.close()

if __name__ == '__main__':
    main()

