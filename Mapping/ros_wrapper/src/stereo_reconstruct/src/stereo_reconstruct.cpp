#include <ros/ros.h>
#include <pluginlib/class_list_macros.h>
#include <nodelet/nodelet.h>

#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>

#include <image_geometry/stereo_camera_model.h>
#include <image_transport/image_transport.h>
#include <image_transport/subscriber_filter.h>

#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/subscriber.h>

#include <cv_bridge/cv_bridge.h>

#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <pcl/pcl_base.h>
#include <pcl/point_types.h>

#include <pcl_conversions/pcl_conversions.h>

#include <pcl/common/common.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d_omp.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/frustum_culling.h>
#include <pcl/filters/random_sample.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>

#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/sac_segmentation.h>

namespace stereo_reconstruct
{

    class StereoReconstruct : public nodelet::Nodelet
    {
        public:
            StereoReconstruct() :
                maxDepth_(0.0),
                minDepth_(0.0),
                voxelSize_(0.0),
                noiseFilterRadius_(0.0),
                noiseFilterMinNeighbors_(5),
                approxSyncStereo_(0),
                exactSyncStereo_(0),
                isMilliMeter_(true),
                isUseOCL_(false),
                frame_id_depth_("stereo_depth_optical_frame"),
                frame_id_cloud_("stereo_cloud_optical_frame"),
                offset_t_(0),
                offset_b_(0),
                pcl_cloud_(new pcl::PointCloud<pcl::PointXYZRGB>),
                depth_frame_(nullptr)
        {}

            virtual ~StereoReconstruct()
            {
                if(approxSyncStereo_)
                    delete approxSyncStereo_;
                if(exactSyncStereo_)
                    delete exactSyncStereo_;
                if(depth_frame_)
                    delete depth_frame_;
            }

        private:
            virtual void onInit()
            {
                ros::NodeHandle & nh  = getNodeHandle();
                ros::NodeHandle & pnh = getPrivateNodeHandle();

                int queueSize = 10;
                bool approxSync = true;

                pnh.param("approx_sync", approxSync, approxSync);
                pnh.param("queue_size", queueSize, queueSize);
                pnh.param("max_depth", maxDepth_, maxDepth_);
                pnh.param("min_depth", minDepth_, minDepth_);
                pnh.param("voxel_size", voxelSize_, voxelSize_);
                pnh.param("noise_filter_radius", noiseFilterRadius_, noiseFilterRadius_);
                pnh.param("noise_filter_min_neighbors", noiseFilterMinNeighbors_, noiseFilterMinNeighbors_);
                pnh.param("isMilliMeter", isMilliMeter_, isMilliMeter_);
                pnh.param("isUseOCL", isUseOCL_, isUseOCL_);
                pnh.param("frame_id_cloud", frame_id_cloud_, frame_id_cloud_);
                pnh.param("frame_id_depth", frame_id_depth_, frame_id_depth_);
                pnh.param("offset_t", offset_t_, offset_t_);
                pnh.param("offset_b", offset_b_, offset_b_);

                NODELET_INFO("Approximate time sync = %s", approxSync?"true":"false");

                if(approxSync) {
                    approxSyncStereo_ = new message_filters::Synchronizer<MyApproxSyncStereoPolicy>(
                            MyApproxSyncStereoPolicy(queueSize), imageLeft_, imageRight_, cameraInfoLeft_, cameraInfoRight_);
                    approxSyncStereo_->registerCallback(boost::bind(&StereoReconstruct::stereoCallback, this, _1, _2, _3, _4));
                }
                else {
                    exactSyncStereo_ = new message_filters::Synchronizer<MyExactSyncStereoPolicy>(
                            MyExactSyncStereoPolicy(queueSize), imageLeft_, imageRight_, cameraInfoLeft_, cameraInfoRight_);
                    exactSyncStereo_->registerCallback(boost::bind(&StereoReconstruct::stereoCallback, this, _1, _2, _3, _4));
                }

                ros::NodeHandle left_nh(nh, "left");
                ros::NodeHandle right_nh(nh, "right");
                ros::NodeHandle left_pnh(pnh, "left");
                ros::NodeHandle right_pnh(pnh, "right");
                image_transport::ImageTransport left_it(left_nh);
                image_transport::ImageTransport right_it(right_nh);
                image_transport::TransportHints hintsLeft("raw", ros::TransportHints(), left_pnh);
                image_transport::TransportHints hintsRight("raw", ros::TransportHints(), right_pnh);

                imageLeft_.subscribe(left_it, left_nh.resolveName("image"), 1, hintsLeft);
                imageRight_.subscribe(right_it, right_nh.resolveName("image"), 1, hintsRight);
                cameraInfoLeft_.subscribe(left_nh, "camera_info", 1);
                cameraInfoRight_.subscribe(right_nh, "camera_info", 1);

                cloudPub_ = nh.advertise<sensor_msgs::PointCloud2>("cloud", 1);

                image_transport::ImageTransport depth_it(nh);
                depthPub_ = depth_it.advertiseCamera("depth_ghc", 1, false);
            }

            void stereoCallback(const sensor_msgs::ImageConstPtr& imageLeft,
                    const sensor_msgs::ImageConstPtr& imageRight,
                    const sensor_msgs::CameraInfoConstPtr& camInfoLeft,
                    const sensor_msgs::CameraInfoConstPtr& camInfoRight)
            {
                if(!(imageLeft->encoding.compare(sensor_msgs::image_encodings::MONO8) == 0 ||
                            imageLeft->encoding.compare(sensor_msgs::image_encodings::MONO16) == 0 ||
                            imageLeft->encoding.compare(sensor_msgs::image_encodings::BGR8) == 0 ||
                            imageLeft->encoding.compare(sensor_msgs::image_encodings::RGB8) == 0) ||
                        !(imageRight->encoding.compare(sensor_msgs::image_encodings::MONO8) == 0 ||
                            imageRight->encoding.compare(sensor_msgs::image_encodings::MONO16) == 0 ||
                            imageRight->encoding.compare(sensor_msgs::image_encodings::BGR8) == 0 ||
                            imageRight->encoding.compare(sensor_msgs::image_encodings::RGB8) == 0))
                {
                    NODELET_ERROR("Input type must be image=mono8,mono16,rgb8,bgr8 (enc=%s)", imageLeft->encoding.c_str());
                    return;
                }

                if(cloudPub_.getNumSubscribers() || depthPub_.getNumSubscribers())
                {
                    cv_bridge::CvImageConstPtr ptrLeftImage  = cv_bridge::toCvShare(imageLeft,  "mono8");
                    cv_bridge::CvImageConstPtr ptrRightImage = cv_bridge::toCvShare(imageRight, "mono8");

                    const cv::Mat &mat_left  = ptrLeftImage->image;
                    const cv::Mat &mat_right = ptrRightImage->image;

                    cv::Rect roi = cv::Rect(0, offset_t_, mat_left.cols, mat_right.rows - offset_t_ - offset_b_);

                    cv::Mat mono_l = cv::Mat(mat_left,  roi);
                    cv::Mat mono_r = cv::Mat(mat_right, roi);

                    image_geometry::StereoCameraModel stereo_camera_model;
                    stereo_camera_model.fromCameraInfo(*camInfoLeft, *camInfoRight);

                    cv::Mat mat_disp;
                    {
                        int blockSize_         = 15;  //15
                        int minDisparity_      = 0;   //0
                        int numDisparities_    = 64;  //64
                        int preFilterSize_     = 9;   //9
                        int preFilterCap_      = 31;  //31
                        int uniquenessRatio_   = 15;  //15
                        int textureThreshold_  = 10;  //10
                        int speckleWindowSize_ = 100; //100
                        int speckleRange_      = 4;   //4

                        if(!isUseOCL_) {
                            cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
                            stereo->setBlockSize(blockSize_);
                            stereo->setMinDisparity(minDisparity_);
                            stereo->setNumDisparities(numDisparities_);
                            stereo->setPreFilterSize(preFilterSize_);
                            stereo->setPreFilterCap(preFilterCap_);
                            stereo->setUniquenessRatio(uniquenessRatio_);
                            stereo->setTextureThreshold(textureThreshold_);
                            stereo->setSpeckleWindowSize(speckleWindowSize_);
                            stereo->setSpeckleRange(speckleRange_);
                            stereo->compute(mono_l, mono_r, mat_disp);

                        } else {
                            cv::ocl::setUseOpenCL(true);

                            cv::UMat uleft, uright, udisp;
                            mono_l.copyTo(uleft);
                            mono_r.copyTo(uright);

                            cv::Ptr<cv::StereoBM> stereo = cv::StereoBM::create();
                            stereo->setBlockSize(blockSize_);
                            stereo->setMinDisparity(minDisparity_);
                            stereo->setNumDisparities(numDisparities_);
                            stereo->setPreFilterSize(preFilterSize_);
                            stereo->setPreFilterCap(preFilterCap_);
                            stereo->setPreFilterType(stereo->PREFILTER_XSOBEL);
                            stereo->setTextureThreshold(0);
                            // stereo->setTextureThreshold(textureThreshold_);
                            stereo->setSpeckleWindowSize(speckleWindowSize_);
                            stereo->setSpeckleRange(speckleRange_);

                            stereo->compute(uleft, uright, udisp);

                            udisp.copyTo(mat_disp);
                        }

                        //                cv::Ptr<cv::StereoMatcher> right_matcher = cv::ximgproc::createRightMatcher(stereo);
                        //                cv::Mat right_disp;
                        //                right_matcher->compute(rightMono, leftMono, right_disp);
                        //
                        //                cv::Mat filtered_disp;
                        //                cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;
                        //                wls_filter = cv::ximgproc::createDisparityWLSFilter(stereo);
                        //                double lambda = 8000.0;
                        //                double sigma = 1.5;
                        //                wls_filter->setLambda(lambda);
                        //                wls_filter->setSigmaColor(sigma);
                        //                wls_filter->filter(disparity, leftImage, filtered_disp, right_disp);
                        //
                        //                cv::Mat filtered_disp_vis;
                        //                cv::ximgproc::getDisparityVis(filtered_disp, filtered_disp_vis, 1.0);
                        //                cv::namedWindow("filtered disparity", WINDOW_AUTOSIZE);
                        //                cv::imshow("filtered disparity", filtered_disp_vis);
                        //                cv::waitKey(10);
                    }

                    pcl_cloud_->height = (uint32_t)mono_l.rows;
                    pcl_cloud_->width  = (uint32_t)mono_l.cols;
                    pcl_cloud_->is_dense = false;
                    pcl_cloud_->resize(pcl_cloud_->height * pcl_cloud_->width);

                    int depth_type = isMilliMeter_ ? CV_16UC1 : CV_32FC1;

                    if(depth_frame_ == nullptr)
                        depth_frame_ = new cv::Mat(mat_left.size(), depth_type);

                    *depth_frame_ = cv::Mat::zeros(mat_left.size(), depth_type);

                    for(int h = 0; h < (int)pcl_cloud_->height; h++) {
                        for (int w = 0; w < (int) pcl_cloud_->width; w++) {

                            pcl::PointXYZRGB &pt = pcl_cloud_->at(h * pcl_cloud_->width + w);

                            unsigned char v = mono_l.at<unsigned char>(h, w);
                            pt.b = v;
                            pt.g = v;
                            pt.r = v;

                            float disp = mat_disp.type() == CV_16SC1 ? float(mat_disp.at<short>(h, w)) / 16.0f : mat_disp.at<float>(h, w);

                            cv::Point3f ptXYZ;
                            {
                                // inspired from ROS image_geometry/src/stereo_camera_model.cpp
                                if (disp > 0.0f && stereo_camera_model.baseline() > 0.0f && stereo_camera_model.left().fx() > 0.0f) {
                                    //Z = baseline * f / (d + cx1-cx0);
                                    float c = 0.0f;
                                    if (stereo_camera_model.right().cx() > 0.0f && stereo_camera_model.left().cx() > 0.0f)
                                        c = stereo_camera_model.right().cx() - stereo_camera_model.left().cx();
                                    
                                    float W = stereo_camera_model.baseline() / (disp + c);

                                    ptXYZ = cv::Point3f(
                                            (cv::Point2f(w, h).x - stereo_camera_model.left().cx()) * W,
                                            (cv::Point2f(w, h).y - stereo_camera_model.left().cy()) * W,
                                            stereo_camera_model.left().fx() * W
                                            );
                                } else {
                                    float bad_point = std::numeric_limits<float>::quiet_NaN();
                                    ptXYZ = cv::Point3f(bad_point, bad_point, bad_point);
                                }
                            }

                            if (std::isfinite(ptXYZ.x) && std::isfinite(ptXYZ.y) && std::isfinite(ptXYZ.z) && ptXYZ.z >= 0) {
                                pt.x = ptXYZ.x;
                                pt.y = ptXYZ.y;
                                pt.z = ptXYZ.z;
                            } else
                                pt.x = pt.y = pt.z = std::numeric_limits<float>::quiet_NaN();

                            float depth = pt.z;

                            if (isMilliMeter_) {
                                unsigned short depthMM = 0;

                                if (depth <= (float) USHRT_MAX) 
                                    depthMM = (unsigned short) depth;

                                depth_frame_->at<unsigned short>(offset_t_+h, w) = depthMM;

                                pt.x /= 1000.0f;
                                pt.y /= 1000.0f;
                                pt.z /= 1000.0f;
                            } else
                                depth_frame_->at<float>(offset_t_+h, w) = depth;
                        }
                    }

                    publishDepth(*depth_frame_, camInfoLeft);
                    publishCloud(pcl_cloud_, imageLeft->header);
                }
            }

            void publishDepth(cv::Mat &depth, const sensor_msgs::CameraInfoConstPtr &camInfo) {

                std::string encoding = "";

                if(depth.type() == CV_16UC1)
                    encoding = sensor_msgs::image_encodings::TYPE_16UC1;
                if(depth.type() == CV_32FC1)
                    encoding = sensor_msgs::image_encodings::TYPE_32FC1;

                sensor_msgs::Image left_img_msg;
                std_msgs::Header header_left;
                header_left.frame_id = frame_id_depth_;
                header_left.stamp = ros::Time::now();
                cv_bridge::CvImage left_img_bridge;
                left_img_bridge = cv_bridge::CvImage(header_left, encoding, depth);
                left_img_bridge.toImageMsg(left_img_msg);

                sensor_msgs::Image left_data;
                left_data.header.frame_id = frame_id_depth_;
                left_data.height = depth.rows;
                left_data.width  = depth.cols ;
                left_data.encoding = encoding;
                left_data.is_bigendian = false;
                left_data.step = depth.step;
                left_data.data = left_img_msg.data;

                sensor_msgs::CameraInfo left_info;
                left_info = *camInfo;
                left_info.header = left_data.header;

                ros::Time stamp = ros::Time::now();

                depthPub_.publish(left_data, left_info, stamp);
            }

            void publishCloud(pcl::PointCloud<pcl::PointXYZRGB>::Ptr & pclCloud, const std_msgs::Header & header)
            {
                if(pclCloud->size() && (minDepth_ != 0.0 || maxDepth_ > minDepth_)) {
                    pcl::PassThrough<pcl::PointXYZRGB> filter;
                    filter.setNegative(false);
                    filter.setFilterFieldName("z");
                    filter.setFilterLimits(minDepth_, maxDepth_ > minDepth_ ? maxDepth_ : std::numeric_limits<float>::max());
                    filter.setInputCloud(pclCloud);
                    filter.filter(*pclCloud);
                }

                if(pclCloud->size() && voxelSize_ > 0.0) {
                    pcl::VoxelGrid<pcl::PointXYZRGB> filter;
                    filter.setLeafSize(voxelSize_, voxelSize_, voxelSize_);
                    filter.setInputCloud(pclCloud);
                    filter.filter(*pclCloud);
                }

                if(pclCloud->empty() && noiseFilterRadius_ > 0.0 && noiseFilterMinNeighbors_ > 0) {
                    if (voxelSize_ <= 0.0 && !(minDepth_ != 0.0 || maxDepth_ > minDepth_)) {
                        std::vector<int> indices;
                        pcl::removeNaNFromPointCloud(*pclCloud, *pclCloud, indices);
                    }

                    pcl::IndicesPtr indices(new std::vector<int>(pclCloud->size()));
                    {
                        pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB>(false));
                        int oi = 0; // output iterator
                        tree->setInputCloud(pclCloud);
                        for (unsigned int i = 0; i < pclCloud->size(); ++i) {
                            std::vector<int> kIndices;
                            std::vector<float> kDistances;
                            int k = tree->radiusSearch(pclCloud->at(i), noiseFilterRadius_, kIndices, kDistances);
                            if (k > noiseFilterMinNeighbors_)
                                indices->at(oi++) = i;

                        }
                        indices->resize(oi);
                    }

                    pcl::PointCloud<pcl::PointXYZRGB>::Ptr tmp(new pcl::PointCloud<pcl::PointXYZRGB>);
                    pcl::copyPointCloud(*pclCloud, *indices, *tmp);
                    pclCloud = tmp;
                }

                sensor_msgs::PointCloud2 rosCloud;
                pcl::toROSMsg(*pclCloud, rosCloud);
                rosCloud.header.stamp = header.stamp;
                rosCloud.header.frame_id = frame_id_cloud_;

                cloudPub_.publish(rosCloud);
            }

        private:
            double maxDepth_;
            double minDepth_;
            double voxelSize_;
            double noiseFilterRadius_;
            bool isMilliMeter_;
            bool isUseOCL_;
            int noiseFilterMinNeighbors_;
            int offset_t_;
            int offset_b_;

            pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud_;
            cv::Mat *depth_frame_;

            ros::Publisher cloudPub_;
            image_transport::CameraPublisher depthPub_;

            image_transport::SubscriberFilter imageLeft_;
            image_transport::SubscriberFilter imageRight_;
            message_filters::Subscriber<sensor_msgs::CameraInfo> cameraInfoLeft_;
            message_filters::Subscriber<sensor_msgs::CameraInfo> cameraInfoRight_;

            typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> MyApproxSyncStereoPolicy;
            message_filters::Synchronizer<MyApproxSyncStereoPolicy> * approxSyncStereo_;

            typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::CameraInfo> MyExactSyncStereoPolicy;
            message_filters::Synchronizer<MyExactSyncStereoPolicy> * exactSyncStereo_;

            std::string frame_id_cloud_;
            std::string frame_id_depth_;
    };

    PLUGINLIB_EXPORT_CLASS(stereo_reconstruct::StereoReconstruct, nodelet::Nodelet);
}

