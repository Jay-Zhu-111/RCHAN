from helper import Helper

helper = Helper()
(hits, AUCs) = helper.evaluate_top()

# Recall
count = [0.0, 0.0, 0.0, 0.0]
for num in hits:
    for i in range(count.__len__()):
        if num[i] == 1:
            count[i] += 1
Recall = []
for i in range(count.__len__()):
    Recall.append(count[i] / hits.__len__())

# AUC
count = 0.0
for num in AUCs:
    count = count + num
AUC = count / AUCs.__len__()

print("Recall@5", Recall[0])
print("Recall@10", Recall[1])
print("Recall@15", Recall[2])
print("Recall@20", Recall[3])
print(AUC)
