import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import Const

dataset =  Const.dataset

class RHAN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, drop_ratio, area_num, cat_num):
        super(RHAN, self).__init__()

        print('embedding size', embedding_dim)
        print('using nine att layer with cat')

        self.itemembeds = ItemEmbeddingLayer(num_items, embedding_dim)
        self.shan_area = MySHAN(num_users, num_items, embedding_dim, drop_ratio, area_num, self.itemembeds).cuda()
        self.shan_cat = MySHAN(num_users, num_items, embedding_dim, drop_ratio, cat_num, self.itemembeds).cuda()
        self.linear = nn.Linear(2 * embedding_dim, embedding_dim)
        with open('{}/cat_dict.txt'.format(dataset), 'r', encoding='UTF-8') as f:
            self.item_cat = eval(f.read())
        with open('{}/area_dict_9.txt'.format(dataset), 'r', encoding='UTF-8') as f:
            self.item_area = eval(f.read())
        self.embedding_dim = embedding_dim
        self.area_num = area_num
        self.cat_num = cat_num

        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                w_range = np.sqrt(3 / embedding_dim)
                nn.init.uniform_(m.weight, -w_range, w_range)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, user_inputs, record_inputs, item_inputs):
        re_user = torch.Tensor().cuda()
        re_item = torch.Tensor().cuda()

        for i in range(user_inputs.__len__()):
            user = user_inputs[i]
            item = [item_inputs[i]]

            record = record_inputs[i]

            # area_input = {}
            # for one_record in record:
            #     one_record = int(one_record)
            #     if one_record != -1:
            #         area_id = self.item_area[one_record]
            #         if area_id in area_input.keys():
            #             area_input[area_id].append(one_record)
            #         else:
            #             area_input[area_id] = [one_record]
            #     else:
            #         break

            cat_input = {}
            for one_record in record:
                one_record = int(one_record)
                if one_record != -1:
                    cat_id = self.item_cat[one_record]
                    if cat_id in cat_input.keys():
                        cat_input[cat_id].append(one_record)
                    else:
                        cat_input[cat_id] = [one_record]
                else:
                    break

            hybrid = self.shan_cat(user, cat_input)

            # # 区域混合表示 1 * K
            # area = self.shan_area(user, area_input)
            # # 类型混合表示 1 * K
            # cat = self.shan_cat(user, cat_input)
            # # 区域—类型连接 1 * 2K
            # area_cat = torch.cat((area, cat), dim=1)
            # # 混合表示 1 * K
            # hybrid = self.linear(area_cat)

            hybrid = torch.reshape(hybrid, (1, 1, self.embedding_dim))
            if re_user.size(0) == 0:
                re_user = hybrid
            else:
                re_user = torch.cat((re_user, hybrid), 0)

            # 得到项目嵌入向量
            item_embed = self.itemembeds(torch.LongTensor(item).cuda())
            item_embed = torch.reshape(item_embed, (1, 1, self.embedding_dim))
            if re_item.size(0) == 0:
                re_item = item_embed
            else:
                re_item = torch.cat((re_item, item_embed), 0)

        re = torch.bmm(re_user, torch.transpose(re_item, 1, 2))
        return re


class MySHAN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, drop_ratio, input_num, item_embeds):
        super(MySHAN, self).__init__()
        self.userembeds = UserEmbeddingLayer(num_users, embedding_dim)
        self.itemembeds = item_embeds

        # init using list
        self.attention_list = []
        # for i in range(input_num):
        #     self.attention_list.append(AttentionLayer(2 * embedding_dim, drop_ratio).cuda())

        self.att0 = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.att1 = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.att2 = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.att3 = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.att4 = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.att5 = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.att6 = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.att7 = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.att8 = AttentionLayer(2 * embedding_dim, drop_ratio)
        for i in range(input_num):
            exec('self.attention_list.append(self.att{})'.format(i))

        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        self.embedding_dim = embedding_dim
        self.input_num = input_num

        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                w_range = np.sqrt(3 / embedding_dim)
                nn.init.uniform_(m.weight, -w_range, w_range)
            if isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.01)

    def forward(self, user, input_dict):
        count_input = 0
        area_all = torch.Tensor().cuda()
        # print('input_dict', input_dict.keys().__len__())
        # print(input_dict.keys())
        # print(self.input_num)

        for i in range(self.input_num):
            input_list_ = input_dict.get(i)
            if input_list_ is None:
                continue
            input_list = []
            for j in range(0, input_list_.__len__()):
                if input_list_[j] != -1:
                    input_list.append(input_list_[j])
            if input_list.__len__() != 0:
                user_numb = []
                for _ in input_list:
                    user_numb.append(user)
                user_embed = self.userembeds(torch.LongTensor(user_numb).cuda())
                # print('user_embed', user_embed.size())
                area_embed = self.itemembeds(torch.LongTensor(input_list).cuda())
                # print('area_embed', area_embed.size())

                # 连接用户和项目嵌入 L * 2K
                user_area_embed = torch.cat((user_embed, area_embed), dim=1)
                # print('user_area_embed', user_area_embed.size())

                # 权重 L * 1
                at_wt = self.attention_list[i](user_area_embed)
                # at_wt = self.att0(user_area_embed)
                # print('wt', at_wt.size())

                # 用户长期表示 1 * K
                u_area = torch.matmul(at_wt, area_embed)
                # print('u_area', u_area.size())

                # 计数
                count_input += 1
                # 区域表示 count_input * K
                if area_all.size(0) == 0:
                    area_all = u_area
                else:
                    area_all = torch.cat((area_all, u_area), 0)
            else:
                print("input_list is empty")
                print("input_list_ len:", input_list_.__len__())

        # 第二层注意力网络
        # 用户嵌入 count_input * K
        user_numb2 = []
        for _ in range(count_input):
            user_numb2.append(user)
        user_embed2 = self.userembeds(torch.LongTensor(user_numb2).cuda())

        # 连接用户和项目嵌入 count_input * 2K
        u_area_all = torch.cat((user_embed2, area_all), dim=1)
        # print('aaa', user_embed2.size())
        # print('bbb', area_all.size())
        # print('u_area_all', u_area_all.size())

        # 权重 (S + 1) * 1
        at_wt2 = self.attention(u_area_all)

        # 用户混合表示 1 * K
        user_hybrid = torch.matmul(at_wt2, area_all)

        return user_hybrid


class UserEmbeddingLayer(nn.Module):
    def __init__(self, num_users, embedding_dim):
        super(UserEmbeddingLayer, self).__init__()
        self.userEmbedding = nn.Embedding(num_users, embedding_dim)
        # torch.nn.init.normal(self.userEmbedding.weight)

    def forward(self, user_inputs):
        user_embeds = self.userEmbedding(user_inputs)
        return user_embeds


class ItemEmbeddingLayer(nn.Module):
    def __init__(self, num_items, embedding_dim):
        super(ItemEmbeddingLayer, self).__init__()
        self.itemEmbedding = nn.Embedding(num_items, embedding_dim)

    def forward(self, item_inputs):
        item_embeds = self.itemEmbedding(item_inputs)
        return item_embeds


class AttentionLayer(nn.Module):
    def __init__(self, embedding_dim, drop_ratio=0):
        super(AttentionLayer, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(embedding_dim, 16),
            nn.ReLU(),
            nn.Dropout(drop_ratio),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        out = self.linear(x)
        weight = F.softmax(out.view(1, -1), dim=1)
        return weight
