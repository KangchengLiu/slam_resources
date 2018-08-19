#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>

using namespace std;
using namespace cv;

void GetPair( Mat &imgL, Mat &imgR, vector<Point2f> &ptsL, vector<Point2f> &ptsR );
void GetPairBM( Mat &imgL, Mat &imgR, vector<Point2f> &ptsL, vector<Point2f> &ptsR );
void ChooseKeyPointsBM( Mat_<float> &disp, int nod, int noe, int nof,
                        vector<Point2f> & ptsL, vector<Point2f> & ptsR );
void FixDisparity( Mat_<float> & disp, int numberOfDisparities );
void CalcDisparity( Mat &imgL, Mat &imgR, Mat_<float> &disp, int nod );
void StereoTo3D( vector<Point2f> ptsL, vector<Point2f> ptsR, vector<Point3f> &pts3D,
                 float focalLenInPixel, float baselineInMM, Mat img,
                 Point3f &center3D, Vec3f &size3D);
void TriSubDiv( vector<Point2f> &pts, Mat &img, vector<Vec3i> &tri );

int main() {

    Mat imgL = imread("../data/view1s.jpg");
    Mat	imgR = imread("../data/view5s.jpg");
    if (!(imgL.data) || !(imgR.data))
    {
        cerr<<"can't load image!"<<endl;
        exit(1);
    }

    imshow("imgL",imgL);
    waitKey(0);

    float stdWidth = 600, resizeScale = 1;
    if (imgL.cols > stdWidth * 1.2)
    {
        resizeScale = stdWidth / imgL.cols;
        Mat imgL1,imgR1;
        resize(imgL, imgL1, Size(), resizeScale, resizeScale);
        resize(imgR, imgR1, Size(), resizeScale, resizeScale);
        imgL = imgL1.clone();
        imgR = imgR1.clone();
    }

    cout<<"calculating feature points..."<<endl;
    vector<Point2f> ptsL, ptsR;
    vector<int> ptNum;
    //GetPair(imgL, imgR, ptsL, ptsR);
    GetPairBM(imgL, imgR, ptsL, ptsR);

    vector<Point3f> pts3D;
    float focalLenInPixel = 3740 * resizeScale, baselineInMM = 160;
    Point3f center3D;
    Vec3f size3D;
    float scale = .2; // scale the z coordinate so that it won't be too large spreaded
    focalLenInPixel *= scale;
    cout<<"calculating 3D coordinates..."<<endl;
    StereoTo3D(ptsL, ptsR, pts3D, focalLenInPixel, baselineInMM, imgL, center3D, size3D);

    cout<<"doing triangulation..."<<endl;
    size_t pairNum = ptsL.size();
    vector<Vec3i> tri;
    TriSubDiv(ptsL, imgL, tri);

    return 0;
}

#define MAXM_FILTER_TH	.8	// threshold used in GetPair
#define HOMO_FILTER_TH	60	// threshold used in GetPair
#define NEAR_FILTER_TH	40	// diff points should have distance more than NEAR_FILTER_TH
void GetPair( Mat &imgL, Mat &imgR, vector<Point2f> &ptsL, vector<Point2f> &ptsR )
{
    Mat descriptorsL, descriptorsR;
    double tt = (double)getTickCount();

    Ptr<FeatureDetector> detector = FeatureDetector::create( "FAST" ); // factory mode
    vector<KeyPoint> keypointsL, keypointsR;
    detector->detect( imgL, keypointsL );
    detector->detect( imgR, keypointsR );

    Ptr<DescriptorExtractor> de = DescriptorExtractor::create("SIFT");
    //SurfDescriptorExtractor de(4,2,true);
    de->compute( imgL, keypointsL, descriptorsL );
    de->compute( imgR, keypointsR, descriptorsR );

    tt = ((double)getTickCount() - tt)/getTickFrequency(); // 620*555 pic, about 2s for SURF, 120s for SIFT

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create( "FlannBased" );
    vector<vector<DMatch>> matches;
    matcher->knnMatch( descriptorsL, descriptorsR, matches, 2 ); // L:query, R:train

    vector<DMatch> passedMatches; // save for drawing
    DMatch m1, m2;
    vector<Point2f> ptsRtemp, ptsLtemp;
    for( size_t i = 0; i < matches.size(); i++ )
    {
        m1 = matches[i][0];
        m2 = matches[i][1];
        if (m1.distance < MAXM_FILTER_TH * m2.distance)
        {
            ptsRtemp.push_back(keypointsR[m1.trainIdx].pt);
            ptsLtemp.push_back(keypointsL[i].pt);
            passedMatches.push_back(m1);
        }
    }

    Mat HLR;
    HLR = findHomography( Mat(ptsLtemp), Mat(ptsRtemp), CV_RANSAC, 3 );
    cout<<"Homography:"<<endl<<HLR<<endl;
    Mat ptsLt;
    perspectiveTransform(Mat(ptsLtemp), ptsLt, HLR);

    vector<char> matchesMask( passedMatches.size(), 0 );
    int cnt = 0;
    for( size_t i1 = 0; i1 < ptsLtemp.size(); i1++ )
    {
        Point2f prjPtR = ptsLt.at<Point2f>((int)i1,0); // prjx = ptsLt.at<float>((int)i1,0), prjy = ptsLt.at<float>((int)i1,1);
        // inlier
        if( abs(ptsRtemp[i1].x - prjPtR.x) < HOMO_FILTER_TH &&
            abs(ptsRtemp[i1].y - prjPtR.y) < 2) // restriction on y is more strict
        {
            vector<Point2f>::iterator iter = ptsL.begin();
            for (;iter!=ptsL.end();iter++)
            {
                Point2f diff = *iter - ptsLtemp[i1];
                float dist = abs(diff.x)+abs(diff.y);
                if (dist < NEAR_FILTER_TH) break;
            }
            if (iter != ptsL.end()) continue;

            ptsL.push_back(ptsLtemp[i1]);
            ptsR.push_back(ptsRtemp[i1]);
            cnt++;
            if (cnt%1 == 0) matchesMask[i1] = 1; // don't want to draw to many matches
        }
    }

    Mat outImg;
    drawMatches(imgL, keypointsL, imgR, keypointsR, passedMatches, outImg,
                Scalar::all(-1), Scalar::all(-1), matchesMask, DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    char title[50];
    sprintf(title, "%.3f s, %d matches, %d passed", tt, matches.size(), cnt);
    imshow(title, outImg);
    waitKey();
}

void GetPairBM( Mat &imgL, Mat &imgR, vector<Point2f> &ptsL, vector<Point2f> &ptsR )
{
    Mat_<float> disp;
    imshow("left image", imgL);

    int numOfDisp = 80; // number of disparity, must be divisible by 16// algorithm parameters that can be modified
    CalcDisparity(imgL, imgR, disp, numOfDisp);
    Mat dispSave, dispS;
    normalize(disp, dispSave, 0, 1, NORM_MINMAX);
    dispSave.convertTo(dispSave, CV_8U, 255);
    imwrite("disp.jpg", dispSave);

    int numOfEgdePt = 80, numOfFlatPt = 50;	// algorithm parameters that can be modified
    ChooseKeyPointsBM(disp, numOfDisp, numOfEgdePt, numOfFlatPt, ptsL, ptsR);
    waitKey();
}

void ChooseKeyPointsBM( Mat_<float> &disp, int nod, int noe, int nof,
                        vector<Point2f> & ptsL, vector<Point2f> & ptsR )
{
    Mat_<float>  dCopy, dx, dy, dEdge;
    dCopy = disp.colRange(Range(nod, disp.cols)).clone();
    normalize(dCopy, dCopy, 0, 1, NORM_MINMAX);

    imshow("disparity", dCopy);
    Mat dShow(dCopy.size(),CV_32FC3);

    if (dCopy.channels() == 1)
        cvtColor(dCopy, dShow, CV_GRAY2RGB);//ￕ￢ﾸ￶ￊ�ﾾ￝ￓ￐ￎￊￌ￢ dshow

    imshow("disparity", dShow);

    int sobelWinSz = 7;// algorithm parameters that can be modified
    Sobel(dCopy, dx, -1, 1, 0, sobelWinSz);
    Sobel(dCopy, dy, -1, 0, 1, sobelWinSz);
    magnitude(dx, dy, dEdge);
    normalize(dEdge, dEdge, 0, 10, NORM_MINMAX);
    imshow("edge of disparity", dEdge);
    waitKey();

    int filterSz[] = {50,30};	// algorithm parameters that can be modified
    float slope[] = {4,8};	// algorithm parameters that can be modified
    int keepBorder = 5;	// algorithm parameters that can be modified
    int cnt = 0;
    double value;
    float minValue = .003;	// algorithm parameters that can be modified
    Point2f selPt1, selPt2;
    Mat_<float> dEdgeCopy1 = dEdge.clone();

    // find the strongest edges, assign 1 or 2 key points near it
    while (cnt < noe)
    {

        Point loc;
        minMaxLoc(dEdgeCopy1, NULL, &value, NULL, &loc);
        if (value < minValue) break;

        float dx1 = dx(loc), dy1 = dy(loc);
        if (abs(dx1) >= abs(dy1))
        {
            selPt1.y = selPt2.y = loc.y;
            selPt1.x = loc.x - (dx1 > 0 ? slope[1] : slope[0]) + nod;
            selPt2.x = loc.x + (dx1 > 0 ? slope[0] : slope[1]) + nod;
            if (selPt1.x > keepBorder+nod)
            {
                ptsL.push_back(selPt1);
                ptsR.push_back(selPt1 - Point2f(disp(selPt1), 0));
                circle(dShow, selPt1-Point2f(nod,0), 2, CV_RGB(255,0,0), 2);
                cnt++;
            }
            if (selPt2.x < disp.cols - keepBorder)
            {
                ptsL.push_back(selPt2);
                ptsR.push_back(selPt2 - Point2f(disp(selPt2), 0));
                circle(dShow, selPt2-Point2f(nod,0), 2, CV_RGB(0,255,0), 2);
                cnt++;
            }

            imshow("disparity",dShow);
            //waitKey();

            int left = min(filterSz[1], loc.x),
                    top = min(filterSz[0], loc.y),
                    right = min(filterSz[1], dCopy.cols-loc.x-1),
                    bot = min(filterSz[0], dCopy.rows-loc.y-1);
            Mat sub = dEdgeCopy1(Range(loc.y-top, loc.y+bot+1), Range(loc.x-left, loc.x+right+1));
            sub.setTo(Scalar(0));
            //imshow("processing disparity edge", dEdgeCopy1);
            //waitKey();
        }
        else
        {
            selPt1.x = selPt2.x = loc.x+nod;
            selPt1.y = loc.y - (dy1 > 0 ? slope[1] : slope[0]);
            selPt2.y = loc.y + (dy1 > 0 ? slope[0] : slope[1]);
            if (selPt1.y > keepBorder)
            {
                ptsL.push_back(selPt1);
                ptsR.push_back(selPt1 - Point2f(disp(selPt1), 0));
                circle(dShow, selPt1-Point2f(nod,0), 2, CV_RGB(255,255,0), 2);
                cnt++;
            }
            if (selPt2.y < disp.rows-keepBorder)
            {
                ptsL.push_back(selPt2);
                ptsR.push_back(selPt2 - Point2f(disp(selPt2), 0));
                circle(dShow, selPt2-Point2f(nod,0), 2, CV_RGB(0,255,255), 2);
                cnt++;
            }

            imshow("disparity",dShow);
            //waitKey();

            int left = min(filterSz[0], loc.x),
                    top = min(filterSz[1], loc.y),
                    right = min(filterSz[0], dCopy.cols-loc.x-1),
                    bot = min(filterSz[1], dCopy.rows-loc.y-1);
            Mat sub = dEdgeCopy1(Range(loc.y-top, loc.y+bot+1), Range(loc.x-left, loc.x+right+1));
            sub.setTo(Scalar(0));
            //imshow("processing disparity edge", dEdgeCopy1);
            //waitKey();
        }

    }

    int filterSz0 = 6;// algorithm parameters that can be modified
    keepBorder = 3;// algorithm parameters that can be modified
    cnt = 0;
    Mat_<float> dEdgeCopy2;// = dEdge.clone();
    GaussianBlur(dEdge, dEdgeCopy2, Size(0,0), 5);
    char str[10];

    // find the flat areas, assign 1 key point near it
    while (cnt < nof)
    {

        Point2i loc;
        minMaxLoc(dEdgeCopy2, &value, NULL, &loc, NULL);
        if (value == 10) break;

        loc.x += nod;
        if (loc.x > keepBorder+nod && loc.y > keepBorder &&
            loc.x < disp.cols && loc.y < disp.rows)
        {
            cv::Point2f loc2f = loc;
            ptsL.push_back(loc);
            ptsR.push_back(loc2f - cv::Point2f(disp(loc), 0));
            circle(dShow, loc2f-Point2f(nod,0), 2, CV_RGB(255,0,255), 2);
            cnt++;
            sprintf(str, "%.1f", disp(loc));
            putText(dShow, str, Point(loc.x-nod+3, loc.y), FONT_HERSHEY_SIMPLEX, .3, CV_RGB(255,0,255));
            imshow("disparity",dShow);
        }

        loc.x -= nod;
        int filterSz1 = (10-value*3)*filterSz0;
        int left = min(filterSz1, loc.x),
                top = min(filterSz1, loc.y),
                right = min(filterSz1, dCopy.cols-loc.x-1),
                bot = min(filterSz1, dCopy.rows-loc.y-1);
        Mat sub = dEdgeCopy2(Range(loc.y-top, loc.y+bot+1), Range(loc.x-left, loc.x+right+1));
        sub.setTo(Scalar(10));
        //imshow("processing disparity flat area", dEdgeCopy2);
    }
}

void CalcDisparity( Mat &imgL, Mat &imgR, Mat_<float> &disp, int nod )
{
    enum { STEREO_BM=0, STEREO_SGBM=1, STEREO_HH=2 };
    int alg = STEREO_SGBM;

    StereoSGBM sgbm;
    int cn = imgR.channels();

    sgbm.SADWindowSize = 3;
    sgbm.numberOfDisparities = nod;
    sgbm.preFilterCap = 63;
    sgbm.P1 = 8*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.P2 = 32*cn*sgbm.SADWindowSize*sgbm.SADWindowSize;
    sgbm.minDisparity = 0;
    sgbm.uniquenessRatio = 10;
    sgbm.speckleWindowSize = 100;
    sgbm.speckleRange = 32;
    sgbm.disp12MaxDiff = 1;
    sgbm.fullDP = alg == STEREO_HH;

    Mat dispTemp, disp8;
    sgbm(imgL, imgR, dispTemp);
    dispTemp.convertTo(disp, CV_32FC1, 1.0/16);
    disp.convertTo(disp8, CV_8U, 255.0/nod);
    imshow("origin disparity", disp8);
    //waitKey();

    FixDisparity(disp, nod);
    disp.convertTo(disp8, CV_8U, 255.0/nod);
    imshow("fixed disparity", disp8);
}

// roughly smooth the glitches on the disparity map
void FixDisparity( Mat_<float> & disp, int numberOfDisparities )
{
    Mat_<float> disp1;
    float lastPixel = 10;
    float minDisparity = 23;// algorithm parameters that can be modified
    for (int i = 0; i < disp.rows; i++)
    {
        for (int j = numberOfDisparities; j < disp.cols; j++)
        {
            if (disp(i,j) <= minDisparity) disp(i,j) = lastPixel;
            else lastPixel = disp(i,j);
        }
    }
    int an = 4;	// algorithm parameters that can be modified
    copyMakeBorder(disp, disp1, an,an,an,an, BORDER_REPLICATE);
    Mat element = getStructuringElement(MORPH_ELLIPSE, Size(an*2+1, an*2+1));
    morphologyEx(disp1, disp1, CV_MOP_OPEN, element);
    morphologyEx(disp1, disp1, CV_MOP_CLOSE, element);
    disp = disp1(Range(an, disp.rows-an), Range(an, disp.cols-an)).clone();
}

// calculate 3d coordinates.
// for rectified stereos: pointLeft.y == pointRight.y
// the origin for both image is the top-left corner of the left image.
// the x-axis points to the right and the y-axis points downward on the image.
// the origin for the 3d real world is the optical center of the left camera
// object -> optical center -> image, the z value decreases.

void StereoTo3D( vector<Point2f> ptsL, vector<Point2f> ptsR, vector<Point3f> &pts3D,
                 float focalLenInPixel, float baselineInMM, Mat img,
                 Point3f &center3D, Vec3f &size3D) // output variable, the center coordinate and the size of the object described by pts3D
{
    vector<Point2f>::iterator iterL = ptsL.begin(), iterR = ptsR.begin();

    float xl, xr, ylr;
    float imgH = float(img.rows), imgW = float(img.cols);
    Point3f pt3D;
    float minX = 1e9, maxX = -1e9;
    float minY = 1e9, maxY = -1e9;
    float minZ = 1e9, maxZ = -1e9;

    Mat imgShow = img.clone();
    char str[100];
    int ptCnt = ptsL.size(), showPtNum = 30, cnt = 0;
    int showIntv = max(ptCnt/showPtNum, 1);
    for ( ; iterL != ptsL.end(); iterL++, iterR++)
    {
        xl = iterL->x;
        xr = iterR->x; // need not add baseline
        ylr = (iterL->y + iterR->y)/2;

        //if (yl-yr>5 || yr-yl>5) // may be wrong correspondence, discard. But vector can't be changed during iteration
        //{}

        pt3D.z = -focalLenInPixel * baselineInMM / (xl-xr); // xl should be larger than xr, if xl is shot by the left camera
        pt3D.y = -(-ylr + imgH/2) * pt3D.z / focalLenInPixel;
        pt3D.x = (imgW/2 - xl) * pt3D.z / focalLenInPixel;

        minX = min(minX, pt3D.x); maxX = max(maxX, pt3D.x);
        minY = min(minY, pt3D.y); maxY = max(maxY, pt3D.y);
        minZ = min(minZ, pt3D.z); maxZ = max(maxZ, pt3D.z);
        pts3D.push_back(pt3D);

        if ((cnt++)%showIntv == 0)
        {
            Scalar color = CV_RGB(rand()&64,rand()&64,rand()&64);
            sprintf(str, "%.0f,%.0f,%.0f", pt3D.x, pt3D.y, pt3D.z);
            putText(imgShow, str, Point(xl-13,ylr-3), FONT_HERSHEY_SIMPLEX, .3, color);
            circle(imgShow, *iterL, 2, color, 3);
        }

    }

    imshow("back project", imgShow);
    waitKey();

    center3D.x = (minX+maxX)/2;
    center3D.y = (minY+maxY)/2;
    center3D.z = (minZ+maxZ)/2;
    size3D[0] = maxX-minX;
    size3D[1] = maxY-minY;
    size3D[2] = maxZ-minZ;
}

// used for doing delaunay trianglation with opencv function
bool isGoodTri( Vec3i &v, vector<Vec3i> & tri )
{
    int a = v[0], b = v[1], c = v[2];
    v[0] = min(a,min(b,c));
    v[2] = max(a,max(b,c));
    v[1] = a+b+c-v[0]-v[2];
    if (v[0] == -1) return false;

    vector<Vec3i>::iterator iter = tri.begin();
    for(;iter!=tri.end();iter++)
    {
        Vec3i &check = *iter;
        if (check[0]==v[0] &&
            check[1]==v[1] &&
            check[2]==v[2])
        {
            break;
        }
    }
    if (iter == tri.end())
    {
        tri.push_back(v);
        return true;
    }
    return false;
}

void TriSubDiv( vector<Point2f> &pts, Mat &img, vector<Vec3i> &tri )
{
    CvSubdiv2D* subdiv;//The subdivision itself // 细分
    CvMemStorage* storage = cvCreateMemStorage(0); ;//Storage for the Delaunay subdivsion //用来存储三角剖分
    Rect rc = Rect(0,0, img.cols, img.rows); //Our outer bounding box //我们的外接边界盒子

    subdiv = cvCreateSubdiv2D( CV_SEQ_KIND_SUBDIV2D, sizeof(*subdiv),
                               sizeof(CvSubdiv2DPoint),
                               sizeof(CvQuadEdge2D),
                               storage );//为数据申请空间

    cvInitSubdivDelaunay2D( subdiv, rc );//rect sets the bounds

    //如果我们的点集不是32位的，在这里我们将其转为CvPoint2D32f，如下两种方法。
    for (size_t i = 0; i < pts.size(); i++)
    {
        CvSubdiv2DPoint *pt = cvSubdivDelaunay2DInsert( subdiv, pts[i] );
        pt->id = i;
    }

    CvSeqReader reader;
    int total = subdiv->edges->total;
    int elem_size = subdiv->edges->elem_size;

    cvStartReadSeq( (CvSeq*)(subdiv->edges), &reader, 0 );
    Point buf[3];
    const Point *pBuf = buf;
    Vec3i verticesIdx;
    Mat imgShow = img.clone();

    srand( (unsigned)time( NULL ) );
    for( int i = 0; i < total; i++ )
    {
        CvQuadEdge2D* edge = (CvQuadEdge2D*)(reader.ptr);

        if( CV_IS_SET_ELEM( edge ))
        {
            CvSubdiv2DEdge t = (CvSubdiv2DEdge)edge;
            int iPointNum = 3;
            Scalar color = CV_RGB(rand()&255,rand()&255,rand()&255);

            //bool isNeg = false;
            int j;
            for(j = 0; j < iPointNum; j++ )
            {
                CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg( t );
                if( !pt ) break;
                buf[j] = pt->pt;
                //if (pt->id == -1) isNeg = true;
                verticesIdx[j] = pt->id;
                t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );
            }
            if (j != iPointNum) continue;
            if (isGoodTri(verticesIdx, tri))
            {
                //tri.push_back(verticesIdx);
                polylines( imgShow, &pBuf, &iPointNum,
                           1, true, color,
                           1, CV_AA, 0);
                //printf("(%d, %d)-(%d, %d)-(%d, %d)\n", buf[0].x, buf[0].y, buf[1].x, buf[1].y, buf[2].x, buf[2].y);
                //printf("%d\t%d\t%d\n", verticesIdx[0], verticesIdx[1], verticesIdx[2]);
                //imshow("Delaunay", imgShow);
                //waitKey();
            }

            t = (CvSubdiv2DEdge)edge+2;

            for(j = 0; j < iPointNum; j++ )
            {
                CvSubdiv2DPoint* pt = cvSubdiv2DEdgeOrg( t );
                if( !pt ) break;
                buf[j] = pt->pt;
                verticesIdx[j] = pt->id;
                t = cvSubdiv2DGetEdge( t, CV_NEXT_AROUND_LEFT );
            }
            if (j != iPointNum) continue;
            if (isGoodTri(verticesIdx, tri))
            {
                //tri.push_back(verticesIdx);
                polylines( imgShow, &pBuf, &iPointNum,
                           1, true, color,
                           1, CV_AA, 0);
                //printf("(%d, %d)-(%d, %d)-(%d, %d)\n", buf[0].x, buf[0].y, buf[1].x, buf[1].y, buf[2].x, buf[2].y);
                //printf("%d\t%d\t%d\n", verticesIdx[0], verticesIdx[1], verticesIdx[2]);
                //imshow("Delaunay", imgShow);
                //waitKey();
            }
        }

        CV_NEXT_SEQ_ELEM( elem_size, reader );

    }

    //RemoveDuplicate(tri);
    char title[100];
    sprintf(title, "Delaunay: %d Triangles", tri.size());
    imshow(title, imgShow);
    waitKey();
}
