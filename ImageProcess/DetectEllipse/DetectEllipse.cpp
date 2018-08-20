#include "DetectEllipse.h"

CDetectEllipse::CDetectEllipse(void)
{
}

CDetectEllipse::~CDetectEllipse(void)
{
}

//��ȡԲ��
void CDetectEllipse::GetPointsCenter(IplImage *src, std::vector<CvPoint2D32f> &imagePoints, std::vector<int> &areaPoints)
{
    // src Ϊ�Ҷ�ͼ�񣬷�����Ҫת��BGR2GRAY 
    // �������
    std::vector<CvPoint2D32f> ptrOnImageFeature;
    std::vector<CvPoint2D32f> ptrOnImage;
    std::vector<CvPoint2D32f> ptrOnPanelFeature;
    std::vector<CvPoint2D32f> ptrOnPanel;
    std::vector<CvPoint2D32f>::iterator piter;

    IplImage* srcbw = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );  // ����һ���Ҷ�ͼ��
    IplImage* gray  = cvCreateImage( cvGetSize(src), IPL_DEPTH_8U, 1 );
    IplImage *dxx   = cvCreateImage( cvGetSize(src), IPL_DEPTH_32F,1);
    IplImage *dyy   = cvCreateImage( cvGetSize(src), IPL_DEPTH_32F,1);

    // ���������˲�
    IplConvKernel* element=0;
    int an = 2;
    element = cvCreateStructuringElementEx(an*2+1,an*2+1,an,an,CV_SHAPE_ELLIPSE,0);

    cvErode(src,srcbw,element,1);
    cvDilate(srcbw,src,element,1);
    double firstThresh = cvMean(src);

    //  cxq  �ͷ�element
    cvReleaseStructuringElement(&element);


    // �����ݶ�
    ImageGradient(src,dxx,dyy,srcbw);  // dx and dy,dx^2+dy^2�� srcbwΪ��ֵͼ��

    cvCopyImage(srcbw,gray);//gray = cvCloneImage(srcbw);

    CvMemStorage* storage = cvCreateMemStorage(0);		// ���������洢����
    CvMemStorage* storageLabel = cvCreateMemStorage(0);
    CvSeq* contours = 0;
    cvFindContours( srcbw, storage, &contours, sizeof(CvContour),	
            CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0) );	// �����������
    //	cvSaveImage(".\\Image\\contour.bmp",srcbw);
    // 	cvSaveImage("dxx.bmp",dxx);
    // 	cvSaveImage("dyy.bmp",dyy);
    //	cvConvert(dxx,gray);
    //	cvNamedWindow("Gradient");
    //	cvShowImage("Gradient",gray);
    //
    /////////////////////////////////////////////////////////////////////////	
    //	cvReleaseImage( &src );
    cvReleaseImage( &srcbw );

    //	����Ѱ�����ĵ�����

    std::vector<double> ptrMoment;
    std::vector<double>::iterator iter;
    CvMoments moments;
    CvPoint2D32f point;
    float axsa = 0.0f;//test��Բ������
    int ct =0;
    for( ; contours != 0; contours = contours->h_next )
    {
        double perimeter = cvArcLength(contours, CV_WHOLE_SEQ, CV_SEQ_FLAG_CLOSED);//�����ܳ�
        int  area = fabs(cvContourArea(contours, CV_WHOLE_SEQ)); 
        double metric = (4 * 3.1415926 * area)/(perimeter * perimeter);

        //if (perimeter > 0 && area >= 20 && contours->total > 6 && metric>0.65) //area <= 3000 &&
        if (perimeter > 0 && area >= 15 && contours->total > 6 && metric>0.45) //area <= 3000 &&//��궨��
            //if (perimeter > 0 && area >= 50 && contours->total > 30 && metric>0.35) //area <= 3000 &&
            //if (perimeter > 0 && area >= 400 && contours->total > 30 && metric>0.35) //area <= 3000 &&
        {
            CvRect box = cvBoundingRect( contours, 0);
            cvMoments( contours, &moments);
            CvSeq* seq = cvCreateSeq(CV_32SC2,sizeof(CvSeq),sizeof(CvPoint),storageLabel);
            CvPoint pt;
            for ( int i=box.y; i<=box.y+box.height;i++)
            {
                for (int j=box.x; j<=box.x+box.width;j++)
                {

                    if (((uchar*)(gray->imageData + gray->widthStep*i))[j]>100)
                    {
                        pt.x = j;
                        pt.y = i;
                        cvSeqPush(seq,&pt);
                    }
                }
            }

            CvBox2D ellipses;
            ellipses = cvFitEllipse2(seq);
            point = ellipses.center;
            axsa = ellipses.size.width/2;//test��Բ������

            int total = seq->total;
            CvPoint2D32f ptf;
            CvSeq* seq2 = cvCreateSeq(CV_32FC2,sizeof(CvSeq),sizeof(CvPoint2D32f),storageLabel);
            float a,b,c;
            for(int k=0;k<total;k++) 
            {			
                cvSeqPopFront(seq,&pt);
                a=((float*)(dxx->imageData + dxx->widthStep*pt.y))[pt.x];
                b=((float*)(dyy->imageData + dyy->widthStep*pt.y))[pt.x];
                // �����������ƶ������µ�Բ�ģ�������ⲻ�ȶ�
                c=-(a*(pt.x-ellipses.center.x)+b*(pt.y-ellipses.center.y));

                if ( abs(c) > 0.0000001)
                {
                    ptf.x = a/c;
                    ptf.y = b/c;
                    cvSeqPush(seq2,&ptf);
                }				
            }
            ellipses = cvFitEllipse2(seq2);
            point.x += ellipses.center.x;
            point.y += ellipses.center.y;
            axsa += ellipses.size.width/2;//test��Բ������
            ////////////////////////2011-10-6//////////////////////////
            int ptty = int(point.y+0.5);
            int pttx = int(point.x+0.5);
            unsigned char tmp1 = src->imageData[ptty*src->width +pttx ];			
            //if(tmp1 > 10 ) //�Ҷ�ֵ����40
            {
                imagePoints.push_back(point);
                areaPoints.push_back(area);
                //axsaList.push_back(axsa);//test��Բ������
                ct++;
            }
            ///////////////////////////////////////////////////////////

            //cxq �ڴ����� seq ��seq2  ����Ϊ��ռ�󣬾ͻ��Զ������ˣ��ڴ�������
            cvClearMemStorage(seq->storage);
            cvClearMemStorage(seq2->storage);

            //			cvReleaseMemStorage(&seq->storage);
            //			cvReleaseMemStorage(&seq2->storage);

        }
    }
    //cxq ����ʹ�ù����ͼ��
    cvReleaseImage(&srcbw);
    cvReleaseImage(&gray);
    cvReleaseImage(&dxx);
    cvReleaseImage(&dyy);


    //cxq   CvMemStorage  CvSeq ����һ���ڴ棬�ͷ�Storage Ҳ���ͷ���CvSeq ����֤��Ŀǰ��������������
    cvReleaseMemStorage(&storage);
    cvReleaseMemStorage(&storageLabel);
}


//����ͼ���ݶ�
void CDetectEllipse::ImageGradient(IplImage *img, IplImage* dx, IplImage* dy, IplImage* dst )
{
    // �ݶȺ˵ļ��㣬����x��y���ݶȾ��ģ��
    float d[5] = {0.2707f, 0.6065f, 0.0f, -0.6065f, -0.2707f};
    float g[5] = {0.1353f, 0.6065f, 1.0f, 0.6065f, 0.1353f};
    CvMat D = cvMat(1, 5, CV_32FC1, &d);
    CvMat G = cvMat(1, 5, CV_32FC1, &g);
    CvMat *T = cvCreateMat(5, 1, CV_32FC1);
    CvMat *opx = cvCreateMat(5,5,CV_32FC1);
    CvMat *opy = cvCreateMat(5,5,CV_32FC1);
    cvTranspose(&G,T);
    cvMatMul(T,&D,opx);
    cvTranspose(&D,T);
    cvMatMul(T,&G,opy);
    IplImage *img32 = cvCreateImage(cvGetSize(img),IPL_DEPTH_32F,1);
    cvThreshold(img,dst,1,0,CV_THRESH_TOZERO);
    cvConvert(dst,img32); 
    IplImage *mask = cvCreateImage(cvGetSize(img),IPL_DEPTH_32F,1);
    IplImage *dx2 = cvCreateImage(cvGetSize(img),IPL_DEPTH_32F,1);
    IplImage *dy2 = cvCreateImage(cvGetSize(img),IPL_DEPTH_32F,1);

    // derivative kernel
    cvFilter2D(img32,dx,opx);
    cvFilter2D(img32,dy,opy);
    cvPow(dx,dx2,2);
    cvPow(dy,dy2,2);

    // �����ݶ� dx^2+dy^2
    cvAdd(dx2,dy2,img32);	
    double dmean = cvMean(img32)*3;   // ȡ��ֵ��3��Ϊ����ֵ���Ƿ�ǡ������ʱ��img32Ϊ�����ݶȡ�
    cvThreshold(img32,mask,dmean,255,CV_THRESH_BINARY);
    cvConvert(mask,dst);


    //	cvSaveImage("edge.bmp",dst);
    cvReleaseImage(&dx2);
    cvReleaseImage(&dy2);
    cvReleaseImage(&img32);
    cvReleaseImage(&mask);

    //cxq
    cvReleaseMat(&T);
    cvReleaseMat(&opx);
    cvReleaseMat(&opy);

}

//���Բ��
int CDetectEllipse::DetectEllipse(IplImage *src, std::vector<CvPoint2D32f>	&centerList)
{
    if(centerList.size() > 0)
    {
        centerList.clear();
        {
            std::vector<CvPoint2D32f>	swapList;
            centerList.swap(swapList);
        }
    }

    // ͼ����Բ�����ĵ���ȡ
    std::vector<int> areaPoints;
    GetPointsCenter(src,centerList, areaPoints);

    std::ofstream fout("circle_saved.txt");
    CvPoint2D32f pt;
    for (unsigned int i=0;i< centerList.size();i++)
    {
        pt = centerList.at(i);
        fout<<i+1<<": "<<pt.x<<","<<pt.y<<std::endl;
    }

    return 0;
}

//��ͼ���ϻ���
void  CDetectEllipse::CrossLine(IplImage *img, std::vector<CvPoint2D32f> imagePoints)
{
    std::vector<CvPoint2D32f>::iterator iter;
    CvPoint pt1,pt2;
    CvPoint offset;
    //CString digit;
    int line_type = CV_AA;
    CvFont font;
    cvInitFont( &font, CV_FONT_HERSHEY_PLAIN, 3, 3, 0.0, 1, line_type );
    /*	cvInitFont( &font, CV_FONT_HERSHEY_PLAIN, 1.5, 2, 0.0, 1, line_type );*/
    offset.x = 9;
    offset.y = 9;
    int sum=0;
    for ( iter = imagePoints.begin(); iter != imagePoints.end(); iter++)
    {
        sum++;

        if ( (int)iter->x !=0 && (int)iter->y !=0 )
        {
            pt1.x = pt2.x = (int)iter->x;
            pt1.y = pt2.y = (int)iter->y;
            // 			digit.Format("%d",sum);
            // 			cvPutText( img, digit, pt1, &font, CV_RGB(255, 0, 0));

            pt1.x = pt1.x - offset.x;
            pt2.x = pt2.x + offset.x;
            //cvLine(img,pt1,pt2,CV_RGB( 0,255, 0),2);
            cvLine(img,pt1,pt2,CV_RGB( 0,255, 0),2);
            pt1.x = pt2.x = (int)iter->x;
            pt1.y = pt2.y = (int)iter->y;
            pt1.y = pt1.y - offset.y;
            pt2.y = pt2.y + offset.y;
            //cvLine(img,pt1,pt2,CV_RGB( 0,255, 0),2);
            cvLine(img,pt1,pt2,CV_RGB( 0,255, 0),2);
        }
    }
}
