from tornado import web, ioloop, httpserver
import base64
import numpy as np
import cv2
from face_recongnition import testfromimage, check_one, check_all
import pymysql
import simplejson as json
import os
import sys


def mylog(logname,logpath):
    import logging
    # 创建一个日志记录器
    log = logging.getLogger("test_logger")
    log.setLevel(logging.INFO)
    # 创建一个日志处理器
    ## 这里需要正确填写路径和文件名，拼成一个字符串，最终生成一个log文件
    logHandler = logging.FileHandler(filename = logpath+"RiskControlDebugging_"+logname+".log")
    ## 设置日志级别
    logHandler.setLevel(logging.INFO)
    # 创建一个日志格式器
    formats = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                datefmt='[%Y/%m/%d %I:%M:%S]')

    # 将日志格式器添加到日志处理器中
    logHandler.setFormatter(formats)
    # 将日志处理器添加到日志记录器中
    log.addHandler(logHandler)
    return log

class MainPageHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        self.render('index.html')

list_name = []
db = pymysql.connect("localhost", "h", "hyl123", "face")

# 签到页面
class QiandaoHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        cursor = db.cursor()
        # 建立临时表，用于判断重复签到
        sql = "CREATE TEMPORARY TABLE `temp_check` ( " \
              "`Id` int(10) unsigned NOT NULL AUTO_INCREMENT," \
              "`stuid` int(11) NOT NULL," \
              "`stuname` varchar(20) NOT NULL," \
              "`time` timestamp NULL DEFAULT NULL," \
              "PRIMARY KEY (`Id`)," \
              "UNIQUE KEY `stuid` (`stuid`)," \
              "UNIQUE KEY `stuname` (`stuname`)" \
              ")"
        logname = 'test'
        logpath = os.getcwd() + "\\"
        logger = mylog(logname, logpath)
        try:
            cursor.execute(sql)
        except:
            logger.exception(sys.exc_info())
            logger.info("Error in dbconfig_file")
        self.render('qiandao.html')

    def post(self, *args, **kwargs):
        # 接受前端传回照片
        img = self.get_argument('face')
        img = base64.b64decode(img.split(',')[-1])
        nparr = np.frombuffer(img, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # 后端处理，并返回前端
        save_path = './checkpoint/face.ckpt'
        name = testfromimage(save_path, img, db)
        list_name.append(name)

        self.render('tanchuang.html', name=list_name[-1])
        list_name.remove(name)

# 弹窗
class TanchuangHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        print("1")

    def post(self, *args, **kwargs):
        print("tanchuang")
        confirm = self.get_argument('confirm')


# 签到记录页面
class JiluHandler(web.RequestHandler):
    def get(self, *args, **kwargs):
        cursor = db.cursor()
        # 建立临时表，用于统计签到总数
        sql = "CREATE TEMPORARY TABLE `temp_check_number` ( " \
              "`Id` int(10) unsigned NOT NULL AUTO_INCREMENT," \
              "`stuid` int(11) NOT NULL," \
              "`stuname` varchar(20) NOT NULL," \
              "`time` timestamp NULL DEFAULT NULL," \
              "`number` int(10) NOT NULL," \
              "PRIMARY KEY (`Id`)" \
              ")"
        logname = 'test'
        logpath = os.getcwd() + "\\"
        logger = mylog(logname, logpath)
        try:
            cursor.execute(sql)
        except:
            logger.exception(sys.exc_info())
            logger.info("Error in dbconfig_file")
        self.render('jilu.html')

    def post(self, *args, **kwargs):
        # 接受前端传回的查询日期
        date = self.get_argument('date')
        # 当未选日期即假设查询日期date值为0，进行查询所有操作
        if date == "0":
            alldate = check_all(db)
            print(alldate)
            data = []
            for i in range(len(alldate)):
                eachdata = {'stuid': alldate[i].split(',')[0], 'stuname': alldate[i].split(',')[1],
                            'time': alldate[i].split(',')[2], 'number': alldate[i].split(',')[3]}
                data.append(eachdata)
            self.write(json.dumps(data))
        else:
            # 查询某一天日期
            year = date.split('-')[0]
            month = date.split('-')[1]
            day = date.split('-')[2]
            alldate = check_one(db, year, month, day)
            print(alldate)
            data = []
            for i in range(len(alldate)):
                eachdata = {'stuid': alldate[i].split(',')[0], 'stuname': alldate[i].split(',')[1],
                            'time': alldate[i].split(',')[2]}
                data.append(eachdata)
            self.write(json.dumps(data))

def main():
    setting = {
        'template_path': 'templates',
        'static_path': 'static'
    }
    application = web.Application([
        (r"/", MainPageHandler),
        (r"/qiandao", QiandaoHandler),
        (r"/jilu", JiluHandler),
        (r"/tanchuang", TanchuangHandler),
    ], **setting)

    http_server = httpserver.HTTPServer(application)
    http_server.listen(8080)
    ioloop.IOLoop.current().start()


if __name__ == '__main__':
    main()
