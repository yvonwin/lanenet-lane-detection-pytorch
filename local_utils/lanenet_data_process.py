import cv2
import numpy as np

class lanenet_data_process():
    def __init__(self):
        """
        """
        pass

    # 数据点直线拟合；此部分采用opencv中的直线拟合形式，由于可能存在杂散噪声的影响采用CV_DIST_L2、CV_DIST_L1、CV_DIST_L12、CV_DIST_FAIR、CV_DIST_WELSCH、CV_DIST_HUBER中
    # CV_DIST_HUBER作为拟合的方式的限制形式
    def houghlane_fit(lane_pts):
        """
        车道线直线拟合
        :param lane_pts:
        :return:
        """
        if not isinstance(lane_pts, np.ndarray):
            lane_pts = np.array(lane_pts, np.float32)

        # x = lane_pts[:, 0]
        # y = lane_pts[:, 1]
        curlane = cv2.fitLine(lane_pts, cv2.DIST_HUBER, 0, 0.01, 0.01)  # 根据这些点拟合直线
        # line的形式是[[cos  a], [sin a], [point_x], [point_y]], 前面两项是有关直线与Y正半轴（这里指的是屏幕坐标系）夹角a的三角函数，后面两项就是所得拟合直线上的一点的横纵坐标。
        # 我们知道一个直线的倾斜角度和它经过的一个点后就可以唯一确定一条直线。
        k = curlane[1] / curlane[0]
        b = curlane[3] - k * curlane[2]
        # 直接将点计算出相应的直线存储
        tmpline = [0, b, curlane[2], curlane[3]]
        return k, b, tmpline  # 表示方式为kx+b=y

    @staticmethod
    def compute_ptslinear(curlinelist):
        ''' #计算车道线的线性程度
        :param curlinelist: 当前的轨道线数据点
        :return:
        '''
        # 1.1首先选择特征点上的2/3部分的点作为拟合的点 剩下1/3的点作为候选区域的特征点
        listnum=len(curlinelist)
        if(listnum>=20):
            lanerail0 = curlinelist[int(listnum*2/ 3):-3]
        else:
            lanerail0 = curlinelist[int(listnum * 2 / 3):]
        lanerail1 = curlinelist[int(listnum / 3):int(listnum*2/ 3)]
        lanerail2= curlinelist[0:int(listnum / 3)]  # 候选区域测试数据
        k0, b0, line0 = lanenet_data_process.houghlane_fit(lanerail0)
        k1, b1, line1 = lanenet_data_process.houghlane_fit(lanerail1)
        k2, b2, line2 = lanenet_data_process.houghlane_fit(lanerail2)
        linetypeflag=False        #初始轨道线型的设置，直线则设置为True，曲线为False
        k=0
        if (abs(k0-k1) < 1) and (abs(k2-k1)<1):  #分段计算直线的斜率，斜率一致则为直线轨道线
            k=(k1+k2+k0)/3
            linetypeflag=True
        else:
            k = (k1 + k2 + k0) / 3
            linetypeflag=False
        return linetypeflag,k

    @staticmethod
    def get_railline(raillanedata):
        '''计算带状的数据边界线数据
        :param raillanedata:当前轨道带状数据
        :param leftok:左右轨道的标识，true左轨道 false右轨道
        :return:轨道带状的外延线段数据点
        '''
        # 数据分装完毕：过滤掉无效点，取有效点进行曲线拟合：取数据中间1/3的点进行3次曲线拟合得到火车轨道线，分别取内侧轨道数据
        railline=[]  # 存放左右轨道的数据点
        num0 = 0
        curleftY_front = 0
        curleftY = 0
        while num0 < len(raillanedata):
            curleftY_front = curleftY
            curleftY = raillanedata[num0][1]
            if (curleftY == curleftY_front):
                break
            railline.append(raillanedata[num0])  # 左侧轨道装载最小值X
            for num1 in range(num0 + 1, len(raillanedata)):
                if (curleftY == raillanedata[num1][1]):
                    pass
                else:
                    break
            num0 = num1
        return railline

    @staticmethod
    def get_raillane_data(lane_data):
        '''
        :param lane_data: 单条轨道区域数据
        :return:
        '''
        curlane = []  # 轨道外侧线
        cursortlaneR,leftline,rightline=[],[],[]
        index_len=[]
        lane_datatemp=[]
        if len(lane_data)>2:
            for i in range(len(lane_data)):
                index_len.append((i,len(lane_data[i])))
            sortindex_len=sorted(index_len,key=lambda x:-x[1])
            lane_datatemp.append(lane_data[sortindex_len[0][0]])
            lane_datatemp.append(lane_data[sortindex_len[1][0]])
        else:
            lane_datatemp=lane_data
        lane_num=len(lane_datatemp)

        if lane_num<2:
           print('railway detects failed!')
        else:    #判断左右车道并且取相应的外侧曲线点
           cursortlane0 = sorted(set(lane_datatemp[0]), key=lambda x:(x[1],x[0]))  # true表示降序排列，默认是false表示升序排列  按照y值升序排列，图像上左上角点为0点
           cursortlane1 = sorted(set(lane_datatemp[1]), key=lambda x:(x[1],x[0]))  # true表示降序排列，默认是false表示升序排列  按照y值升序排列，图像上左上角点为0点

           #判断左右轨道线，且存储轨道线的方式按照从左到右的方式存储
           if cursortlane0[-1][0]>cursortlane1[-1][0]:
               cursortlaneR = sorted(set(cursortlane0), key=lambda x:(x[1],-x[0]))  # true表示降序排列，x降序排列，右侧轨道取外侧
               leftline=lanenet_data_process.get_railline(cursortlane1)
               rightline=lanenet_data_process.get_railline(cursortlaneR)

           else:
               cursortlaneR = sorted(set(cursortlane1), key=lambda x:(x[1],-x[0]))  # true表示降序排列，x降序排列，右侧轨道取外侧
               leftline = lanenet_data_process.get_railline(cursortlane0)
               rightline = lanenet_data_process.get_railline(cursortlaneR)

           curlane.append(leftline)
           curlane.append(rightline)

        return curlane

    # 3次多项式拟合
    def lane_fit(lane_pts):
        """
        车道线多项式拟合
        :param lane_pts:
        :return:
        """
        if not isinstance(lane_pts, np.ndarray):
            lane_pts = np.array(lane_pts, np.float32)

        x = lane_pts[:, 0]
        y = lane_pts[:, 1]
        f1 = np.polyfit(y, x, 3)  # f1为多项式拟合的系数，此处为3次多项式  最高次幂3，得到4个系数,从高次到低次排列
        p1 = np.poly1d(f1)  # 将系数代入方程，得到函式p1
        return p1, f1  # 表示方式为ay3+by2+cy+d=x

    def get_railwaylines(tmpleftline,tmprightline,linetypeflag,leftk,rightk,Hmin,Hmax):
        '''求解火车轨道线的数据
        :param tmprightline:左侧轨道线
        :param linetypeflag:右侧轨道线
        :param imgW:原始图像的宽度
        :param imgH:原始图像的高度
        :return:火车轨道线左右车道的数据点以及消影点
        '''
        # 根据线性获取左右车道线以及交点
        railleft_pts, railright_pts = [], []  #装载左右道线数据点
        if True == linetypeflag:  # 确认火车轨道的线性为直道
            left1 = (int(tmpleftline[-1][0]), int(tmpleftline[-1][1]))  # 左侧数据切片的最有一个值
            right1 = (int(tmprightline[-1][0]), int(tmprightline[-1][1]))  # 右侧数据切片的最后一个值

            #此处不取消影点作为远端截至点，取图像的上端点部分作为上半部分的截至点
            if leftk==0:
                lefttop0 = (left1[0], Hmin)
                lefttop1 = (left1[0], Hmax)
            else:
                lefttop0 = (left1[0] - int((left1[1] - Hmin) / leftk), Hmin)
                lefttop1 = (left1[0] - int((left1[1] - Hmax) / leftk), Hmax)

            if rightk==0:
                righttop0 = (right1[0], Hmin)
                righttop1 = (right1[0], Hmax)
            else:
                righttop0 = (right1[0] - int((right1[1] - Hmin) / rightk), Hmin)
                righttop1 = (right1[0] - int((right1[1] - Hmax) / rightk), Hmax)

            railleft_pts.append(lefttop0)  # 将两条直线的两个端点分别装载至左右车道轨迹数据列表中
            railleft_pts.append(lefttop1)
            railright_pts.append(righttop0)
            railright_pts.append(righttop1)

        else:  # 确认火车轨道的线性为弯道了，进行3次多项式拟合
            # 将所获得轨道点进行3次多项式拟合
            pleft, pl = lanenet_data_process.lane_fit(tmpleftline)
            pright, pr = lanenet_data_process.lane_fit(tmprightline)
            y_min = Hmin      #目前图像的有效轨道区域占图像的一半
            y_max = Hmax
            y_fit = []
            for i in range(y_min, y_max + 1):
                y_fit.append(i)
            xleft_fit = pleft(y_fit)
            xright_fit = pright(y_fit)
            for j in range(len(y_fit)):
                railleft_pts.append((xleft_fit[j], y_fit[j]))  # 三次拟合的曲线点
                railright_pts.append((xright_fit[j], y_fit[j]))  # 三次拟合的曲线点

        return railleft_pts, railright_pts
