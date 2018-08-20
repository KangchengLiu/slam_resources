#include "DetectEllipse.h"
#include <iostream>

int main()
{
    IplImage *grayImageOut = cvLoadImage("../images/timg.jpeg", CV_LOAD_IMAGE_GRAYSCALE);

    std::vector<CvPoint2D32f> vidiconCenterList;
    CDetectEllipse::DetectEllipse(grayImageOut, vidiconCenterList);

    CDetectEllipse::CrossLine(grayImageOut,vidiconCenterList);
    cvSaveImage("result.bmp",grayImageOut);
    cvReleaseImage(&grayImageOut);

    std::cout<<"Detect Ellipse Done!"<<std::endl;

    return 0;
}
