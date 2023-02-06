train_label = []

for i in range(50000):
    if i < 10000:
        train_label.append(0)
    if i >= 10000 and i < 20000:
        train_label.append(1)
    if i >= 20000 and i < 30000:
        train_label.append(2)
    if i >= 30000 and i < 40000:
        train_label.append(3)
    if i >= 40000 and i < 50000:
        train_label.append(4)
print(len(train_label))
class_indices = [[] for i in range(10)]
print(len(class_indices))
for i, v in enumerate(train_label):
    class_indices[v].append(i)

print("second label 3rd element: ",class_indices[1][3])
print("i: ",i)
print("v: ",v)
