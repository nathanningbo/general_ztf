import os
import shutil
import re
# path = 'D:\\BigTruck\\new\\'
# newpath ='D:\\BigTruck\\new2\\'
#1.rootpath为数据根目录，要求原始图像放在rootpath/raw下
#2.rootpath下不允许有images路径，以及train.txt文件
#3.rootpath/raw下各种类别的文件名需要按照数字排序好
#4.rootpath/images为图像输出路径，同时rootpath/images下会生成相应的label文件
def dataproces(rootpath , mode = True):
    if (not os.path.exists(rootpath)):
        print('rootpath error')
        return 0
    RawImgRootPath = os.path.join(rootpath,'rawvalidate')
    if (not os.path.exists(RawImgRootPath)):
        print('RawImgRootPath error')
        return 0
    ImgRootPath = os.path.join(rootpath,'validate')
    startnum =1
    if (not os.path.exists( ImgRootPath)):
        os.mkdir( ImgRootPath);
    for file in os.listdir(RawImgRootPath):
        rawimgtypespath = os.path.join(RawImgRootPath, file)
        imgtypespath = os.path.join( ImgRootPath, file)
        if os.path.isfile(rawimgtypespath) == False:
            startnum = images_copy_and_rename(rawimgtypespath, imgtypespath, '.jpg', startnum)
    direct_gen_text(ImgRootPath ,rootpath)

def images_copy_and_rename(oldpath, newpath ,imagetype , imgstartnum = 1):
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
    print( '路径下有'+ str(count)+'张'+imagetype)
    return count
def gen_img_label_text( imgpath, outfilepath, outfilename ,labelnum):
    f = open(os.path.join(outfilepath, outfilename), 'w')
    for filename in os.listdir(imgpath):
        f.write(filename + labelnum+'\n')
    f.close()

#生成txt文件
def direct_gen_text(imgrootpath ,txtpath, txtname = 'train.txt' ,imgtype = '.jpg'):
    count = 1 #每循环完一次之后，图片id的初始位置
    type = 0  #label的值
    labeltypes = count_img_type_num(imgrootpath)
    f =  open(os.path.join(txtpath, txtname), 'w')
    for labelnum in labeltypes:
        for  n in range(count , count+labelnum):
            f.write(str( n ) + imgtype  + ' ' + str(type) +' '+ '\n')
        type += 1
        count += labelnum
    f.close()

#返回每种类型图片的个数
def count_img_type_num(imgrootpath):
    merget_count = []
    for fn in os.listdir(imgrootpath):  # fn 表示的是文件名
        count = 0
        path = os.path.join(imgrootpath ,str(fn))
        for each in os.listdir(path):
            count += 1
        merget_count.append(count)
    return merget_count

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
        #newpath = 'E://data//train//2'
        #newpath = 'E://data'
        #images_copy_and_rename(path, newpath, '.jpg',12279)
        #gen_img_label_text( path, newpath ,'11112.txt', ' 1 ')
        #direct_gen_text( '.jpg', [ 12278,24174], newpath, 'train.txt' )
        #direct_gen_text('E://data//images', 'E://data//' )
        #data_dict = get_imgpath_and_label(newpath, '9222.txt')
        #print(data_dict)
        #print(count_img_type_num('E://data//images'))
        dataproces('E://data//')