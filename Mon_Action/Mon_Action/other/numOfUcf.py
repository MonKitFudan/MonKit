import os
path = '/home/fair/Desktop/xzf/tsn/PrimatesInLab'
count = 0
for lists in os.listdir(path):
    sub_path = os.path.join(path, lists)
    if os.path.isdir(sub_path):
        count += 1
print(count)