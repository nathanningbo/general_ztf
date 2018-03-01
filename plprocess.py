import os
import shutil
import re
# path = 'D:\\BigTruck\\new\\'
# newpath ='D:\\BigTruck\\new2\\'
def images_copy_and_rename(oldpath, newpath ,imagetype, imgstartnum):
    count = imgstartnum
    if(not os.path.exists(oldpath)):
        return 0
    if (not os.path.exists(newpath)):
        os.mkdir(newpath);
    for file in os.listdir(oldpath):
        if os.path.isfile(os.path.join(oldpath, file))==True:
            if os.path.splitext(file)[1] == imagetype:#.jpg
                newname = str(count) +imagetype
                shutil.copyfile(os.path.join(oldpath, file), os.path.join(newpath, newname))
                count += 1
                print( file )
    print( '路径下有'+ str(count)+'张'+imagetype)
    return 1
def gen_img_label_text( imgpath, outfilepath, outfilename ,labelnum):
    f = open(os.path.join(outfilepath, outfilename), 'w')
    for filename in os.listdir(imgpath):
        f.write(filename + labelnum+'\n')
    f.close()
# imgtype 图片类型 如 '.jpg'
# labeltypes 包括每种样本的数量以及label的种类，如 [10 ,20 ,30],意思是3中type。类型一的图片有10张，类型2的图片有20张
# txtpath , txtname 如名字
# 结果如下：1.jpg 1   2.jpg 1   .....7个....10.jpg 1    #以下换行已省略
#           11.jpg 2   12.jpg 2 .....17个....30.jpg 2
#           31.jpg 3   32.jpg 3 .....27个....60.jpg 3
def direct_gen_text( imgtype, labeltypes ,txtpath, txtname):
    count = 1 #每循环完一次之后，图片id的初始位置
    type = 1  #label的值
    f =  open(os.path.join(txtpath, txtname), 'w')
    for labelnum in labeltypes:
        for  n in range(count , count+labelnum):
            f.write(str( n ) + imgtype  + ' ' + str(type) +' '+ '\n')
        type += 1
        count += labelnum
    f.close()
def get_imgpath_and_label( txtpath, txtname):
    f = open(os.path.join(txtpath, txtname), 'r')
    lines = f.readlines()
    data_dict = dict()
    for line in lines:
        wholedata = re.split(' ', line)
        data_dict[wholedata[0]] = wholedata[1]
    f.close()
    return data_dict

if(__name__ == '__main__'):
        #newpath ='D:\\BigTruck\\new7\\'
        #path = 'E://data//images//2'
        #newpath = 'E://data//train//2'
        newpath = 'E://data'
        #images_copy_and_rename(path, newpath, '.jpg',12279)
        #gen_img_label_text( path, newpath ,'11112.txt', ' 1 ')
        direct_gen_text( '.jpg', [ 12278,24174], newpath, 'train.txt' )
        #data_dict = get_imgpath_and_label(newpath, '9222.txt')
        #print(data_dict)