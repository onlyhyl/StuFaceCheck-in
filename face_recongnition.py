import os
import logging as log
import numpy as np
import tensorflow as tf
import cv2
import cnn as myconv


def createdir(*args):
    """
    新建文件夹
    :param args: 文件夹名称
    :return: None
    """
    for item in args:
        if not os.path.exists(item):
            os.makedirs(item)


IMGSIZE = 64


def getpaddingSize(shape):
    """
    设图像边界填充
    :param shape:图像形状
    :return:填充后的图像矩阵
    """
    h, w = shape
    longest = max(h, w)
    result = (np.array([longest] * 4, int) - np.array([h, h, w, w], int)) // 2
    return result.tolist()


def dealwithimage(img, h=64, w=64):
    """
    处理图像尺寸
    :param img: 图像
    :param h: 高
    :param w: 宽
    :return: 尺寸处理后的图像
    """
    top, bottom, left, right = getpaddingSize(img.shape[0:2])
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    img = cv2.resize(img, (h, w))
    return img


def relight(imgsrc, alpha=1, bias=0):
    """
    处理曝光及过暗图像
    :param imgsrc: 图像
    :param alpha:倍数
    :param bias:偏移量
    :return:亮度处理后的图像
    """
    imgsrc = imgsrc.astype(float)
    imgsrc = imgsrc * alpha + bias
    imgsrc[imgsrc < 0] = 0
    imgsrc[imgsrc > 255] = 255
    imgsrc = imgsrc.astype(np.uint8)
    return imgsrc


def getface(imgpath, outdir):
    """
    把所有处理后的人脸照片写入新文件夹
    :param imgpath:原数据人脸
    :param outdir:所有处理后写入的文件夹名
    :return:None
    """
    filename = os.path.splitext(os.path.basename(imgpath))[0]
    img = cv2.imread(imgpath)
    haar = cv2.CascadeClassifier(
        'C:/Users/Administrator/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = haar.detectMultiScale(gray_img, 1.3, 5)
    n = 0
    for x, y, w, h in faces:
        n += 1
        face = img[y:y + h, x:x + w]
        # may be do not need resize now
        # face = cv2.resize(face, (64, 64))
        face = dealwithimage(face, IMGSIZE, IMGSIZE)
        for inx, (alpha, bias) in enumerate([[1, 1], [1, 50], [0.5, 0]]):
            facetemp = relight(face, alpha, bias)
            cv2.imwrite(os.path.join(outdir, '%s_%d_%d.jpg' % (filename, n, inx)), facetemp)



def getfilesinpath(filedir):
    """
    读取各个人的存放文件夹里面的照片
    :param filedir: 文件夹的路径
    :return: 文件夹中照片完整路径
    """
    for (path, dirnames, filenames) in os.walk(filedir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                yield os.path.join(path, filename)
        for diritem in dirnames:
            getfilesinpath(os.path.join(path, diritem))


def generateface(pairdirs):
    """
    判断是否存在新文件夹，存储所有处理后的图像到新文件夹
    :param pairdirs:存储图像数据的根目录
    :return:None
    """
    for inputdir, outputdir in pairdirs:
        for name in os.listdir(inputdir):
            inputname, outputname = os.path.join(inputdir, name), os.path.join(outputdir, name)
            if os.path.isdir(inputname):
                createdir(outputname)
                for fileitem in getfilesinpath(inputname):
                    getface(fileitem, outputname)


def readimage(pairpathlabel):
    """
    用opencv读取、转化完整路径的照片，转化为二进制矩阵存放在列表
    :param pairpathlabel: 标签文件夹的路径
    :return: img、label作为训练样本
    """
    imgs = []
    labels = []
    for filepath, label in pairpathlabel:
        for fileitem in getfilesinpath(filepath):
            img = cv2.imread(fileitem)
            imgs.append(img)
            labels.append(label)
    return np.array(imgs), np.array(labels)


def onehot(numlist):
    """
    标签转化为onehot编码
    :param numlist: 标签列表
    :return:标签矩阵
    """
    b = np.zeros([len(numlist), max(numlist) + 1])
    b[np.arange(len(numlist)), numlist] = 1
    return b.tolist()


def getfileandlabel(filedir):
    """
    找到存放照片的文件夹，统一标记标签；
    :param filedir: 新文件夹根目录
    :return: 各个人的存放文件夹，和其对应标签
    """
    dictdir = dict([[name, os.path.join(filedir, name)]
                    for name in os.listdir(filedir) if os.path.isdir(os.path.join(filedir, name))])
    # for (path, dirnames, _) in os.walk(filedir) for dirname in dirnames])

    dirnamelist, dirpathlist = dictdir.keys(), dictdir.values()
    indexlist = list(range(len(dirnamelist)))

    return list(zip(dirpathlist, onehot(indexlist))), dict(zip(indexlist, dirnamelist))


def main(_):
    ''' main '''
    savepath = './checkpoint/face.ckpt'
    isneedtrain = False
    if os.path.exists(savepath + '.meta') is False:
        isneedtrain = True
    if isneedtrain:
        # first generate all face
        log.debug('generateface')
        generateface([['./image/trainfaces', './image/trainimages']])
        pathlabelpair, indextoname = getfileandlabel('./image/trainimages')
        train_x, train_y = readimage(pathlabelpair)
        train_x = train_x.astype(np.float32) / 255.0
        log.debug('len of train_x : %s', train_x.shape)
        myconv.train(train_x, train_y, savepath)
        log.debug('training is over, please run again')
    else:
        testfromcamera(savepath)


def testfromcamera(chkpoint):
    """
    调用摄像头进行识别
    :param chkpoint:模型存储路径
    :return:None
    """
    camera = cv2.VideoCapture(0)
    haar = cv2.CascadeClassifier('C:/Users/Administrator/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_alt2.xml')
    pathlabelpair, indextoname = getfileandlabel('./image/trainfaces')
    output = myconv.cnnLayer(len(pathlabelpair))
    predict = output

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, chkpoint)

        n = 1
        while 1:
            if (n <= 20000):
                print('It`s processing %s image.' % n)
                # 读帧
                success, img = camera.read()

                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = haar.detectMultiScale(gray_img, 1.3, 5)
                for x, y, w, h in faces:
                    face = img[y:y + h, x:x + w]
                    face = cv2.resize(face, (IMGSIZE, IMGSIZE))
                    test_x = np.array([face])
                    test_x = test_x.astype(np.float32) / 255.0

                    res = sess.run([predict, tf.argmax(output, 1)],
                                   feed_dict={myconv.x_data: test_x,
                                              myconv.keep_prob_5: 1.0, myconv.keep_prob_75: 1.0})
                    print(res)
                    if len(res):
                        print(indextoname[res[1][0]])

                    cv2.putText(img, indextoname[res[1][0]], (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, 255,
                                2)  # 显示名字
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    n += 1
                cv2.imshow('img', img)
                key = cv2.waitKey(30) & 0xff
                if key == 27:
                    break
            else:
                break
    camera.release()
    cv2.destroyAllWindows()


def testfromimage(chkpoint, img, db):
    """
    图像识别
    :param chkpoint: 模型存储路径
    :param img: 图像
    :param db: 数据库连接对象
    :return: 签到结果
    """
    haar = cv2.CascadeClassifier(
        'C:/Users/Administrator/AppData/Local/Programs/Python/Python37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
    pathlabelpair, indextoname = getfileandlabel('./image/trainfaces')
    output = myconv.cnnLayer(len(pathlabelpair))
    predict = output

    saver = tf.Graph()
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.import_meta_graph('./checkpoint/face.ckpt.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./checkpoint'))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = haar.detectMultiScale(img, 1.3, 5)
        if faces:
            for x, y, w, h in faces:
                face = img[y:y + h, x:x + w]
                face = cv2.resize(face, (IMGSIZE, IMGSIZE))
                test_x = np.array([face])
                test_x = test_x.astype(np.float32) / 255.0
                res = sess.run([predict, tf.argmax(output, 1)],
                               feed_dict={myconv.x_data: test_x,
                                          myconv.keep_prob_5: 1.0, myconv.keep_prob_75: 1.0})
                if len(res):
                    name = indextoname[res[1][0]]
                    print(name)
                    stuname, stuid = seltable(db, name)
                    if stuname == name:
                        if tempcheck(db, stuid, stuname):
                            result = "姓名：%s 学号：%d，签到成功！" % (name, stuid)
                            return result
                        else:
                            return "您已签到成功，请勿重复签到！"
                    db.close()

        else:
            message = "未识别人脸，请重新签到"
            return message


def seltable(db, name):
    """
    学生信息表查找识别人信息
    :param db: 数据库连接
    :param name: 识别人名字
    :return: 学生名字和学生学号
    """
    cursor = db.cursor()
    sql = "select * from stu where stuname = '%s'" % name
    cursor.execute(sql)
    data = cursor.fetchall()
    for row in data:
        stuid = row[1]
        stuname = row[2]
        return stuname, stuid


def tempcheck(db, stuid, stuname):
    """
        识别人信息录入签到表，判断是否重复签到
        :param db:数据连接
        :param stuid:学生学号
        :param stuname:学生姓名
        :return:不重复则录入并返回True，重复则返回False
        """
    cursor = db.cursor()

    # 识别人信息插入临时表，验证是否重复签到
    sql = "insert into temp_check(stuid, stuname, time) values(%d, '%s', now())" % (stuid, stuname)
    try:
        cursor.execute(sql)
        db.commit()
    except:
        print("重复")
        return False

    # 从临时表插入签到表保存签到记录
    sql = "select * from temp_check"
    cursor.execute(sql)
    data = cursor.fetchall()
    for row in data:
        stuid = row[1]
        stuname = row[2]
        time = row[3]
    sql = "insert into check_in(stuid, stuname, time) values(%d, '%s', '%s')" % (stuid, stuname, time)
    cursor.execute(sql)
    db.commit()
    return True


def check_one(db, year, month, day):
    """
    查询某一天签到总人数
    :param db: 数据库连接对象
    :param year: 年
    :param month: 月
    :param day: 日
    :return: 某一天的所有签到数据列表
    """
    cursor = db.cursor()
    sql = " select * from check_in where date_format(time,'%%Y-%%m-%%d')='%s-%s-%s'" % (year, month, day)
    cursor.execute(sql)
    data = cursor.fetchall()
    all = []
    for row in data:
        stuid = row[1]
        stuname = row[2]
        time = row[3]
        all.append(str(stuid) + "," + stuname + "," + str(time))
    return all


def check_all(db):
    """
    查询签到总人数,按签到人数从多到少排序
    :param db: 数据库连接
    :return:所有签到记录数据列表
    """
    cursor = db.cursor()
    sql = " select * from check_in "
    cursor.execute(sql)
    data = cursor.fetchall()
    allid = []
    for row in data:
        stuid = row[1]
        stuname = row[2]
        time = row[3]
        if stuid not in allid:
            allid.append(stuid)
            sql = "select count(stuid) from check_in where stuid=%d" % stuid
            cursor.execute(sql)
            data = cursor.fetchall()
            for row in data:
                number = row[0]
                sql = "insert into temp_check_number(stuid, stuname, time, number) values(%d, '%s', '%s', '%d')" % (
                stuid, stuname, time, number)
                cursor.execute(sql)
                db.commit()

    sql = " select * from temp_check_number order by number DESC"
    cursor.execute(sql)
    data = cursor.fetchall()
    alljilu = []
    for row in data:
        stuid = row[1]
        stuname = row[2]
        time = row[3]
        number = row[4]
        alljilu.append(str(stuid) + "," + stuname + "," + str(time) + "," + str(number))
    return alljilu


if __name__ == '__main__':
    main(0)
    # savepath = './checkpoint/face.ckpt'
    # img = cv2.imread('./image/trainfaces/heyonglin/46.jpg')
    # print(testfromimage(savepath, img, 1))
