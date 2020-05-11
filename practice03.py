#
# path1 = r'D:\E\document\datas\megaage_asian\list\train_name.txt'
# path2 = r'D:\E\document\datas\megaage_asian\list\train_age.txt'
# path3 = r'D:\E\document\datas\megaage_asian\train'
# path4 = r'D:\E\document\datas\megaage_asian\list\train_label.txt'
#
# file = open(path4, "w")
# with open(path1) as line1:
#     with open(path2) as line2:
#         for (name, age) in zip(line1, line2):
#             # print(name)
#             strs1 = str(name).strip('\n')
#             print(strs1)
#             strs2 = str(age).strip('\n')
#             print(strs2)
#             # img_path = os.path.join(path3, "{}".format(strs1[0]))
#             file.write("{} {}\n".format(strs1, strs2))
#             file.flush()
#
#
#

path1 = r'D:\E\document\datas\megaage_asian\list\train_name.txt'
path2 = r'D:\E\document\datas\megaage_asian\list\train_age.txt'
path3 = r'D:\E\document\datas\megaage_asian\train'
path4 = r'D:\E\document\datas\megaage_asian\list\train_label.txt'

file = open(path4, "w")
with open(path1) as line1:
    with open(path2) as line2:
        for (name, age) in zip(line1, line2):
            # print(name)
            strs1 = str(name).strip('\n')
            print(strs1)
            strs2 = str(age).strip('\n')
            print(strs2)
            # img_path = os.path.join(path3, "{}".format(strs1[0]))
            file.write("{} {}\n".format(strs1, strs2))
            file.flush()

