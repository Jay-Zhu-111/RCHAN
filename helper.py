import torch
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import math
import heapq
import Const
dataset = Const.dataset

def clean_data(list_input):
    newList = []
    for i in range(0, list_input.__len__()):
        inList = []
        for j in range(0, list_input[i].__len__()):
            if list_input[i][j] != -1:
                inList.append(list_input[i][j])
        newList.append(inList)
    return newList


step_count = 0
steps = []
losses = []
def save_loss(loss):
    global step_count, steps, losses
    steps.append(step_count)
    step_count = step_count + 1
    losses.append(loss)


def draw_loss():
    # data_loss = np.loadtxt("foursquare/SHAN_loss.txt")
    # x = data_loss[:, 0]
    # y = data_loss[:, 1]
    x = steps
    y = list(losses)
    fig = plt.figure(figsize=(7, 5))
    p2 = pl.plot(x, y, 'r-', label=u'SHAN')
    pl.legend()
    pl.xlabel(u'iters')
    pl.ylabel(u'loss')
    plt.title('Loss')
    pl.show()


class Helper:
    def __init__(self):
        with open('{}/Input/user_input_test.txt'.format(dataset), 'r', encoding='UTF-8') as f:
            self.user_input = eval(f.read())
        with open('{}/Input/item_input_test.txt'.format(dataset), 'r', encoding='UTF-8') as f:
            self.record_input = eval(f.read())

        # i = 0
        # while True:
        #     try:
        #         f = open('foursquare/My/area' + str(i) + '_input_test.txt',  'r', encoding='UTF-8')
        #     except IOError:
        #         break
        #     else:
        #         self.input_list.append(eval(f.read()))
        #         f.close()
        #     i += 1

        with open('{}/Input/pos_item_input_test.txt'.format(dataset), 'r', encoding='UTF-8') as f:
            self.j_input = eval(f.read())
        with open('{}/Input/neg_item_input_test.txt'.format(dataset), 'r', encoding='UTF-8') as f:
            self.k_input = eval(f.read())

        print("验证数据长度", self.user_input.__len__())

    def evaluate_top(self):
        hits, AUCs = [], []
        for idx in range(self.user_input.__len__()):
            (hr, AUC) = eval_one_rating_top(self.user_input[idx],
                                        self.record_input[idx],
                                        self.j_input[idx],
                                        self.k_input[idx])
            hits.append(hr)
            for auc in AUC:
                AUCs.append(auc)
        return hits, AUCs

    def evaluate_model(self, model):
        hits, AUCs = [], []
        for idx in range(self.user_input.__len__()):
            (hr, AUC) = eval_one_rating(model, self.user_input[idx],
                                        self.record_input[idx],
                                        self.j_input[idx],
                                        self.k_input[idx])
            hits.append(hr)
            for auc in AUC:
                AUCs.append(auc)
        return hits, AUCs

    def evaluate_model_with_size(self, model, size):
        hits, AUCs = [], []
        for idx in range(size):
            (hr, AUC) = eval_one_rating(model, self.user_input[idx],
                                        self.record_input[idx],
                                        self.j_input[idx],
                                        self.k_input[idx])
            hits.append(hr)
            for auc in AUC:
                AUCs.append(auc)
        return hits, AUCs


def eval_one_rating_top(u, record,
                    j, k_list):
    k_list.append(j)
    map_item_score = {}
    for i in range(k_list.__len__()):
        item = k_list[i]
        map_item_score[item] = item
    k_list.pop()

    # Evaluate top rank list
    ranklist_5 = heapq.nsmallest(5, map_item_score, key=map_item_score.get)
    ranklist_10 = heapq.nsmallest(10, map_item_score, key=map_item_score.get)
    ranklist_15 = heapq.nsmallest(15, map_item_score, key=map_item_score.get)
    ranklist_20 = heapq.nsmallest(20, map_item_score, key=map_item_score.get)
    hr = []
    hr.append(getHitRatio(ranklist_5, j))
    hr.append(getHitRatio(ranklist_10, j))
    hr.append(getHitRatio(ranklist_15, j))
    hr.append(getHitRatio(ranklist_20, j))
    # hr = getHitRatio(ranklist, j)
    AUC = getAUC(map_item_score, j, k_list)
    return hr, AUC


def eval_one_rating(model, u, record,
                    j, k_list):
    k_list.append(j)
    # Get prediction scores
    map_item_score = {}
    user_var = []
    record_var = []
    for i in range(0, k_list.__len__()):
        user_var.append(u)
        record_var.append(record)

    # input_list_var = []
    # for area in input_list:
    #     li = []
    #     for i in range(k_list.__len__()):
    #         li.append(area)
    #     input_list_var.append(torch.LongTensor(li).cuda())

    user_var = torch.LongTensor(user_var)
    record_var = torch.LongTensor(record_var)
    item_var = torch.LongTensor(k_list)

    predictions = model(user_var, record_var, item_var).cpu()
    for i in range(k_list.__len__()):
        item = k_list[i]
        map_item_score[item] = predictions.data.numpy()[i]
    k_list.pop()

    # Evaluate top rank list
    ranklist_5 = heapq.nlargest(5, map_item_score, key=map_item_score.get)
    ranklist_10 = heapq.nlargest(10, map_item_score, key=map_item_score.get)
    ranklist_15 = heapq.nlargest(15, map_item_score, key=map_item_score.get)
    ranklist_20 = heapq.nlargest(20, map_item_score, key=map_item_score.get)
    hr = []
    hr.append(getHitRatio(ranklist_5, j))
    hr.append(getHitRatio(ranklist_10, j))
    hr.append(getHitRatio(ranklist_15, j))
    hr.append(getHitRatio(ranklist_20, j))
    # hr = getHitRatio(ranklist, j)
    AUC = getAUC(map_item_score, j, k_list)
    return hr, AUC


def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0


def getAUC(map_item_score, j, k_list):
    AUC = []
    j_score = map_item_score[j]
    for item in k_list:
        if map_item_score[item] < j_score:
            AUC.append(1)
        elif map_item_score[item] == j_score:
            AUC.append(0.5)
        else:
            AUC.append(0)
    return AUC


def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

# 测试helper文件
# embedding_size = Const.embedding_size
# drop_ratio = Const.drop_ratio
# epoch = Const.epoch
# batch_size = Const.batch_size
#
# foursquare = Data()
# num_users = foursquare.get_user_size()
# num_items = foursquare.get_item_size()
# shan = SHAN(num_users, num_items, embedding_size, drop_ratio)
# (hits, ndcgs) = evaluate_model(shan)
# print(hits, ndcgs)
