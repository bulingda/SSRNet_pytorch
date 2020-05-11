import os
from PIL import Image


path = r'D:\E\document\datas\megaage_asian\list\train_label.txt'
path1 = r'D:\E\document\datas\face_age_dataset\train'
path2 = r'D:\E\document\datas\megaage_asian\train'

# 把me图片放到face图片中
datas = []
datas.extend(open(path).readlines())


for _, data in enumerate(datas):   # me图片
    # print(data.split()[0])
    for doc_name in os.listdir(path1):  # FACE图片
        # print(doc_name)
        images = data.split()[0]
        labels = data.split()[1]
        print(str(labels.zfill(3)))
        if str(labels.zfill(3)) == str(doc_name):
            img = Image.open(os.path.join(path2, images))  # 取me图片
            # print(img)
            img.save(os.path.join(path1, doc_name, images))
        else:
            print('没遇到一样的？')
