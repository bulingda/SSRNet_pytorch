import os
path = r'D:\E\document\datas\megaage_asian\list'
label_txt = open(os.path.join(path, 'train_name.txt'), "w")
for i in range(1, 80001):
    label_txt.write("{}.jpg\n".format(i))