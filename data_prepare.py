# ------------------------------------------------#
#   进行训练前需要利用这个文件生成cls_train.txt
# ------------------------------------------------#
import os
from os import getcwd

wd = getcwd()
datasets_path = "datasets/"
name_label = os.listdir(datasets_path)
name_label = sorted(name_label)
list_file = open('cls_train.txt', 'w')
i = 0
for label_id, label_name in enumerate(name_label):
    if i < 100:
        img_path = os.path.join(datasets_path, label_name)
        if not os.path.isdir(img_path):
            continue
        photos_name = os.listdir(img_path)
        i = i + 1
        j = 0
        for photo_name in photos_name:
            if j < 10:
                list_file.write(str(label_id) + ";" + '%s/%s' % (wd, os.path.join(img_path, photo_name)))
                list_file.write('\n')
                j = j + 1
list_file.close()
