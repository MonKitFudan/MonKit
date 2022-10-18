import os
import glob
path = "/home1/lyr/CODE/DATA/test_monkey"

trianpath = "/home1/lyr/CODE/DATA/file_list/monkey_train_list.txt"

testpath = "/home1/lyr/CODE/DATA/file_list/monkey_val_list.txt"

cls_index = {
              'Chase':'0',
              'Contact':'1',
              'Hug':'2',
}
f1 = open(trianpath,'a+')
f2 = open(testpath,'a+')
count = 0
servPath='/home1/lyr/CODE/DATA/monkey_frame/'
if (os.path.exists(path)):
    clslist = os.listdir(path)
    # clslist.remove('.DS_Store')
    for cls in clslist:
        avi_files = sorted(glob.glob(path +'/'+ cls + '/*.avi'))
        for avi in avi_files:
            print(avi)
            end_pos = avi.rfind('/') - 1
            start_pos = avi.rfind('/', 0, end_pos)
            avi_name = avi[start_pos + 1:].split('.')[0]
            print(avi_name)
            search_file = servPath+'/'+avi_name
            avi_frames = int(len(os.listdir(search_file)))

            if count%6==0:
                f2.write(servPath+avi_name+' '+ str(avi_frames)+' '+cls_index[cls]+'\n')
            else:
                f1.write(servPath+avi_name + ' ' + str(avi_frames) + ' ' + cls_index[cls] + '\n')
            count += 1
