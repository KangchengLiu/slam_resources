#pragma once

#include <vector>
#include <opencv2/core/types_c.h>
#include <opencv2/core/core_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/legacy/compat.hpp>
#include <opencv2/highgui/highgui_c.h>

#include <fstream>

class CDetectEllipse
{
    public:
        CDetectEllipse(void);
        ~CDetectEllipse(void);	
    public:
        static int	DetectEllipse(IplImage *src, std::vector<CvPoint2D32f>	&centerList);//检测圆心
        static void CrossLine(IplImage *img, std::vector<CvPoint2D32f> imagePoints);//向图像上画线
    private:
        static void GetPointsCenter(IplImage *src, std::vector<CvPoint2D32f> &imagePoints,std::vector<int> &areaPoints);//提取圆心函数
        static void ImageGradient(IplImage *img, IplImage* dx, IplImage* dy, IplImage* dst );//计算图像梯度
};

