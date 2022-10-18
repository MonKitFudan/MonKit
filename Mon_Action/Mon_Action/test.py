import glob
import os
# path_file_number=glob.glob('D:/case/test/testcase/checkdata/*.py')#或者指定文件下个数
# path_file_number=glob.glob(pathname='/home1/xiaozf/dt/fisher_vector_pil14_feature/*.txt') #获取当前文件夹下个数
# print(path_file_number)
# print(len(path_file_number))
file_path = '/home/fair/Desktop/xzf/UCF101/ucf101_frames'
dir_count = 0
file_count = 0
for root, dirs, filenames in os.walk(file_path):
    for dir in dirs:
        dir_count += 1
    # for file in filenames:
    #     file_count += 1
print ('dir_count ', dir_count)
print ('file_/count ', file_count)