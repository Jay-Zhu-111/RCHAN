import json

with open(r'C:\Users\CrazyCat\Desktop\categories.txt', 'r', encoding='UTF-8') as f:
    data = json.load(f)
# print(type(foursquare))
# print(foursquare)

# for item in foursquare.items():
# #     # print(type(item[1]))
# #     # print(item[1])
# #     for item2 in item[1].items():
# #         print(item2)

print(type(data['response']['categories']))
print(data['response']['categories'])
li = data['response']['categories']
for item in li:
    print(item['name'])
    if item['name'] == 'Food':
        re = item

re_list = []
print(re['categories'])
for item in re['categories']:
    re_list.append(item['id'])
    print(item['name'])

with open('foursquare/food.txt', 'w', encoding='UTF-8') as f:
    f.write(str(re_list))
