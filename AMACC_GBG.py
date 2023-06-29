from collections import Counter

import numpy as np

class GranularBall:
    #Some basic attributes of GB
    def __init__(self, data) -> None:
        self.data = data
        self.center, self.radius = calCenterRadius(data[:, 1:])
        self.num = len(data)
        self.label, self.purity = calLabelPurity(data)


#Calculate the label and purity of GB
def calLabelPurity(data):
    if len(data) > 1:
        count = Counter(data[:, 0])
        label = max(count, key=count.get) 
        purity = count[label] / len(data)
    else:
        label = data[0][0]
        purity = 1.0
    return label, purity


#Calculate the center and radius of GB
def calCenterRadius(data):
    center = np.mean(data, axis=0)
    dis = calculateDist(data, center)
    radius = np.mean(dis)
    return center, radius

#Calculate the Euclidean distance between objects
def calculateDist(A, B, flag=0):
    if (flag == 0):
        return np.sqrt(np.sum((A - B)**2, axis=1))
    else:
        return np.sqrt(np.sum((A - B)**2))



#Constructing clusters formed by the majority of samples in data
def generateOmegaCluster(data):
    cluster = []
    todo_data = []
    count = Counter(data[:, 0])
    label = max(count, key=count.get)  
    Omega_Group = data[data[:, 0] == label]
    center = np.mean(Omega_Group[:, 1:], axis=0)
    dis = np.around(calculateDist(data[:, 1:], center), 6)
    Omega_Group_dis = dis[data[:, 0] == label]
    radius = np.around(np.mean(Omega_Group_dis), 6)
    for i in range(len(data)):
        if dis[i] <= radius:
            cluster.append(data[i])
        else:
            todo_data.append(data[i])
    cluster = np.array(cluster)
    todo_data = np.array(todo_data)
    return cluster, todo_data


# Eliminate conflicting relationships：merging heterogeneous nested GBs
def removeConflicts(ball_list):
    gb_remove_tmp = []
    for i in range(len(ball_list) - 1):
        if ball_list[i] in gb_remove_tmp:
            continue
        for j in range(i + 1, len(ball_list)):
            if (ball_list[j] in gb_remove_tmp):
                continue
            if (ball_list[i].label != ball_list[j].label) and (calculateDist(
                    ball_list[i].center, ball_list[j].center,
                    flag=1) <= abs(ball_list[i].radius - ball_list[j].radius)):
                ball_list[j].data = np.concatenate(
                    (ball_list[j].data, ball_list[i].data))
                ball_list[j].label, ball_list[j].purity = calLabelPurity(
                    ball_list[j].data)
                ball_list[j].center, ball_list[j].radius = calCenterRadius(
                    ball_list[j].data[:, 1:])
                gb_remove_tmp.append(ball_list[i])
                break
    for ball in set(gb_remove_tmp):
        ball_list.remove(ball)
    return ball_list

#For any GB, iteratively split it
# Returns the GranularBall class equipped with basic attributes
def splitGB(gb):
    todo_data = gb
    label_num = len(np.unique(gb[:, 0]))
    todo_data_num = len(todo_data)
    ball_list = []
    ball_list_new = []
    while label_num != todo_data_num:
        cluster, todo_data_tmp = generateOmegaCluster(todo_data)
        if len(cluster) > 1:  #去掉离群点构成粒球
            new_ball = GranularBall(cluster)
            ball_list.append(new_ball)
        todo_data = todo_data_tmp
        todo_data_num = len(todo_data)
        if todo_data_num > 1:
            label_num = len(np.unique(todo_data[:, 0]))
        else:
            break
    if len(ball_list) > 1:  #Eliminate conflicts
        ball_list_new = removeConflicts(ball_list)
    return ball_list_new


# Iteratively construct GBs that meet threshold conditions for the entire training dataset
def generateGBList(data, purity):
    GB_List = [GranularBall(data)] 
    i = 0  #cursor
    GB_num = len(GB_List)
    while True:
        if GB_List[i].purity < purity:
            new_split_gbs = splitGB(GB_List[i].data)  
            if len(new_split_gbs) > 1:  
                GB_List[i] = new_split_gbs[0]
                GB_List.extend(new_split_gbs[1:])
            elif len(new_split_gbs) == 1 and (len(new_split_gbs[0].data)
                                              == len(GB_List[i].data)):
                GB_List.pop(i)
            elif len(new_split_gbs) == 1:
                GB_List.pop(i)
                GB_List.append(new_split_gbs[0])
            else:
                GB_List.pop(i)
            GB_num = len(GB_List)
        else:
            i += 1
        if i == GB_num:
            break
    return GB_List





