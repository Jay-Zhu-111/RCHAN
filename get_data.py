import Const
import random
import torch
from  torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader

dataset = Const.dataset
timeStep = Const.time_step
currentTime = Const.current_time


# class subDataset(Dataset):
#     def __init__(self, user, pos, neg):
#         self.user = user
#         # self.list = list
#         self.pos = pos
#         self.neg = neg
#
#     def __len__(self):
#         return len(self.user)
#
#     def __getitem__(self, index):
#         user = torch.LongTensor(self.user[index])
#         # input_list = []
#         # for item in self.list:
#         #     input_list.append(item[index])
#         pos = torch.LongTensor(self.pos[index])
#         neg = torch.LongTensor(self.neg[index])
#         return user, pos, neg


class Data:
    def __init__(self):
        with open('{}/item_set.txt'.format(dataset), 'r', encoding='UTF-8') as item_set_f:
            self.item_set = eval(item_set_f.read())

        with open('{}/user_set.txt'.format(dataset), 'r', encoding='UTF-8') as user_set_f:
            self.user_set = eval(user_set_f.read())

        # with open('foursquare/Input/user_input_train.txt', 'r', encoding='UTF-8') as f:
        #     self.user_input = eval(f.read())
        # with open('foursquare/Input/item_input_train.txt', 'r', encoding='UTF-8') as f:
        #     self.record_input = eval(f.read())
        # with open('foursquare/Input/pos_item_input_train.txt', 'r', encoding='UTF-8') as f:
        #     self.j_input = eval(f.read())
        # with open('foursquare/Input/neg_item_input_train.txt', 'r', encoding='UTF-8') as f:
        #     self.k_input = eval(f.read())

        with open('{}/Input/user_input_train.txt'.format(dataset), 'r', encoding='UTF-8') as f:
            self.user_input = eval(f.read())
        with open('{}/Input/item_input_train.txt'.format(dataset), 'r', encoding='UTF-8') as f:
            self.record_input = eval(f.read())
        with open('{}/Input/pos_item_input_train.txt'.format(dataset), 'r', encoding='UTF-8') as f:
            self.j_input = eval(f.read())
        with open('{}/Input/neg_item_input_train.txt'.format(dataset), 'r', encoding='UTF-8') as f:
            self.k_input = eval(f.read())

        # area_input_list = []
        # for i in range(9):
        #     exec('area_input_list.append(self.area{}_input)'.format(i))
        # self.record_input = []
        # for i in range(self.user_input.__len__()):
        #     record = []
        #     for area_input in area_input_list:
        #         area = area_input[i]
        #         for item in area:
        #             item = int(item)
        #             if item != -1:
        #                 record.append(item)
        #     self.record_input.append(record)
        #
        # L_max = 0
        # for i in range(0, self.record_input.__len__()):
        #     if self.record_input[i].__len__() > L_max:
        #         L_max = self.record_input[i].__len__()
        # for i in range(0, self.record_input.__len__()):
        #     while self.record_input[i].__len__() < L_max:
        #         self.record_input[i].append(-1)

        print("训练数据长度", self.user_input.__len__())
        print("用户数", self.get_user_size())
        print("项目数", self.get_item_size())
        # self.device = device
        # # user = input("输入用户ID:")
        # self.u = '6756'
        # for i in range(0, self.user_set.__len__()):
        #     if self.user_set[i] == self.u:
        #         self.u_id = i
        #         break
        #
        # self.user_info = self.user_dict[self.u]
        #
        # self.S = []
        # self.L = []
        # for item in self.user_info.items():  # 分别存入用户长期和短期项目集
        #     if int(item[0]) > currentTime - timeStep:
        #         self.S.append(item[1])
        #     else:
        #         self.L.append(item[1])

    # 总用户数
    def get_user_size(self):
        return self.user_set.__len__()

    # 总项目数
    def get_item_size(self):
        return self.item_set.__len__()

    # 返回物品
    def get_single_item(self, item_id):
        return self.item_set[item_id]

    # 返回物品序号
    def get_single_item_id(self, item):
        re = 0
        for i in range(0, self.item_set.__len__()):
            if item == self.item_set[i]:
                re = i
                break
        return re

    # 返回物品序号集
    def get_item_id(self, input_set):
        item_list = []
        for item in input_set:
            for i in range(0, self.item_set.__len__()):
                if item == self.item_set[i]:
                    item_list.append(i)
                    break
        return item_list

    # # 返回用户ID
    # def get_u(self):
    #     return self.u_id
    #
    # # 返回长期项目集
    # def get_L(self):
    #     return self.L
    #
    # # 返回短期项目集
    # def get_S(self):
    #     return self.S

    # # 获取观察样本集
    # def get_ob_set(self):
    #     ob_set = []
    #     for item in self.user_info.items():
    #         (u, L, S, j) = self.get_ob(int(item[0]))
    #         ob_set.append((u, L, S, j))
    #     return ob_set

    # 获取一个观察样本
    def get_ob(self, size):
        user_input, L_input, S_input, pos_item_input, neg_item_input = [], [], [], [], []
        for i in range(0, size):
            u = self.user_set[i]
            u_id = i
            user_info = self.user_dict[u]
            for time in user_info.keys():
                L = []
                S = []
                j = 0
                for item in user_info.items():  # 分别存入用户长期和短期项目集
                    if int(item[0]) >= int(time):
                        j = self.get_single_item_id(item[1])
                    elif int(item[0]) > int(time) - timeStep:
                        S.append(self.get_single_item_id(item[1]))
                    else:
                        L.append(self.get_single_item_id(item[1]))
                if L.__len__() == 0 and S.__len__() == 0:
                    continue
                k = self.get_unob(L, S)
                user_input.append(u_id)
                L_input.append(L)
                S_input.append(S)
                pos_item_input.append(j)
                neg_item_input.append(k)
        # L_S_pi_ni = [[L, S, pi, ni] for L, S, pi, ni in zip(L_input, S_input, pos_item_input, neg_item_input)]
        # L_S_pi_ni111 = torch.LongTensor(L_S_pi_ni)
        # print("user_long_tensor", L_S_pi_ni111)
        # print(type(L_S_pi_ni111))

        # 填充L_input
        L_max = 0
        for i in range(0, L_input.__len__()):
            if L_input[i].__len__() > L_max:
                L_max = L_input[i].__len__()
        for i in range(0, L_input.__len__()):
            while L_input[i].__len__() < L_max:
                L_input[i].append(-1)

        # 填充S_input
        S_max = 0
        for i in range(0, S_input.__len__()):
            if S_input[i].__len__() > S_max:
                S_max = S_input[i].__len__()
        for i in range(0, S_input.__len__()):
            while S_input[i].__len__() < S_max:
                S_input[i].append(-1)
        # print(user_input.__len__())
        # tensorU = torch.LongTensor(L_input)
        # print('tensorU')
        # print(tensorU)
        # print(tensorU.size())
        return user_input, L_input, S_input, pos_item_input, neg_item_input

    # 随机获取一个负样本
    def get_unob(self, L, S):
        ob_item = []
        for item in L:
            ob_item.append(item)
        for item in S:
            ob_item.append(item)
        while True:
            k = random.randint(0, self.item_set.__len__() - 1)
            if k not in ob_item:
                break
        return k

    # 返回训练dataloader
    def get_dataloader(self, batch_size):
        # if is_training:
        # with open('foursquare/user_input_train.txt', 'r', encoding='UTF-8') as f:
        #     user = eval(f.read())
        # with open('foursquare/L_input_train.txt', 'r', encoding='UTF-8') as f:
        #     L = eval(f.read())
        # with open('foursquare/S_input_train.txt', 'r', encoding='UTF-8') as f:
        #     S = eval(f.read())
        # with open('foursquare/pos_item_input_train.txt', 'r', encoding='UTF-8') as f:
        #     j = eval(f.read())
        # with open('foursquare/neg_item_input_train.txt', 'r', encoding='UTF-8') as f:
        #     k = eval(f.read())
        # else:
        #     with open('foursquare/user_input_test.txt', 'r', encoding='UTF-8') as f:
        #         user = eval(f.read())
        #     with open('foursquare/L_input_test.txt', 'r', encoding='UTF-8') as f:
        #         L = eval(f.read())
        #     with open('foursquare/S_input_test.txt', 'r', encoding='UTF-8') as f:
        #         S = eval(f.read())
        #     with open('foursquare/pos_item_input_test.txt', 'r', encoding='UTF-8') as f:
        #         j = eval(f.read())
        #     with open('foursquare/neg_item_input_test.txt', 'r', encoding='UTF-8') as f:
        #         k = eval(f.read())

        data = TensorDataset(
            torch.LongTensor(self.user_input),
            torch.LongTensor(self.record_input),
            torch.LongTensor(self.j_input),
            torch.LongTensor(self.k_input)
        )
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        return data_loader

    def get_dataloader_with_size(self, batch_size, size):
        # user, L, S, j, k = self.get_ob(size)
        # print(L)
        data = TensorDataset(
            torch.LongTensor(self.user_input[0: size]),
            torch.LongTensor(self.area0_input[0: size]),
            torch.LongTensor(self.area1_input[0: size]),
            torch.LongTensor(self.area2_input[0: size]),
            torch.LongTensor(self.area3_input[0: size]),
            torch.LongTensor(self.area4_input[0: size]),
            torch.LongTensor(self.area5_input[0: size]),
            torch.LongTensor(self.area6_input[0: size]),
            torch.LongTensor(self.area7_input[0: size]),
            torch.LongTensor(self.area8_input[0: size]),
            torch.LongTensor(self.j_input[0: size]),
            torch.LongTensor(self.k_input[0: size])
        )
        data_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        return data_loader

    # # 返回dataframe
    # def get_dataframe(self):
    #     foursquare = pd.DataFrame.from_dict(self.user_info, orient='index')
    #     foursquare.rename(columns = {'user', 'item'})
    #     return foursquare
    #
    # # 返回dataset
    # def read_data(self, batch_size, is_training):
    #     user_dict = {}
    #     user_dict['time'] = np.array( list(self.user_info.keys()) )
    #     user_dict['item'] = np.array(list(self.user_info.values()))
    #     dataset = tf.foursquare.Dataset.from_tensor_slices(user_dict)
    #     if is_training:
    #         dataset = dataset.shuffle(10000)
    #     dataset = dataset.batch(batch_size)
    #
    #     return dataset

# foursquare = Data()
