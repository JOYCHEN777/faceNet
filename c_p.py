import glob
import os.path
import numpy as np
import os

DATA_PATH = r'/Users/joycyan/Desktop/code/facenet-pytorch-my/lfw'


def create_image_lists():
    print("f1")
    matched_result = set()
    k = 0
    sub_dirs = [x[0] for x in os.walk(DATA_PATH)]
    while len(matched_result) < 300:
        for sub_dir in sub_dirs[1:]:
            extensions = 'jpg'
            file_list = []
            dir_name = os.path.basename(sub_dir)
            file_glob = os.path.join(DATA_PATH, dir_name, '*.' + extensions)
            file_list.extend(glob.glob(file_glob))
            if not file_list: continue
            label_name = dir_name
            length = len(file_list)
            random_number1 = np.random.randint(50)
            random_number2 = np.random.randint(50)
            base_name1 = os.path.basename(file_list[random_number1 % length])
            base_name1 = base_name1[len(label_name) + 4:-4]
            base_name2 = os.path.basename(file_list[random_number2 % length])
            base_name2 = base_name2[len(label_name) + 4:-4]
            if (file_list[random_number1 % length] != file_list[random_number2 % length]):
                matched_result.add(label_name + '\t' + base_name1 + '\t' + base_name2)
                k = k + 1
    return matched_result, k


print("peidui")


def create_pairs():
    unmatched_result = set()
    k = 0
    sds = [x[0] for x in os.walk(DATA_PATH)]
    for sd in sds[1:]:
        extensions = ['jpg']
        file_list = []
        dir_name = os.path.basename(sd)
        for extension in extensions:
            file_glob = os.path.join(DATA_PATH, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))

    length_of_dir = len(sds)
    for j in range(24):
        for i in range(length_of_dir):
            class1 = sds[i]
            class2 = sds[(length_of_dir - i + j - 1) % length_of_dir]
            if ((length_of_dir - i + j - 1) % length_of_dir):
                class1_name = os.path.basename(class1)
                class2_name = os.path.basename(class2)

                extensions = 'jpg'
                file_list1 = []
                file_list2 = []
                file_glob1 = os.path.join(DATA_PATH, class1_name, '*.' + extension)
                file_list1.extend(glob.glob(file_glob1))
                file_glob2 = os.path.join(DATA_PATH, class2_name, '*.' + extension)
                file_list2.extend(glob.glob(file_glob2))
                if file_list1 and file_list2:
                    base_name1 = os.path.basename(file_list1[j % len(file_list1)])
                    base_name1 = base_name1[len(class1_name) + 4:-4]
                    base_name2 = os.path.basename(file_list2[j % len(file_list2)])
                    base_name2 = base_name2[len(class2_name) + 4:-4]

                    s = class2_name + '\t' + base_name2 + '\t' + class1_name + '\t' + base_name1
                    if (s not in unmatched_result):
                        unmatched_result.add(class1_name + '\t' + base_name1 + '\t' + class2_name + '\t' + base_name2)
                    k = k + 1
    return unmatched_result, k


result, k1 = create_image_lists()
print(len(result))

result_un, k2 = create_pairs()
print(len(result_un))

file = open(r'/model_data/pairs.txt', 'w')

result1 = list(result)
result2 = list(result_un)

file.write('10 30\n')

j = 0
for i in range(10):
    j = 0
    for pair in result1[i * 30:i * 30 + 30]:
        j = j + 1
        print(str(j) + ': ' + pair)
        file.write(pair + '\n')
    for pair in result2[i * 30:i * 30 + 30]:
        j = j + 1
        print(str(j) + ': ' + pair)
        file.write(pair + '\n')
