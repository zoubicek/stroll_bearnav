#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <iostream>
#include <algorithm>
#include <stroll_bearnav/FeatureArray.h>
#include <stroll_bearnav/Feature.h>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d.hpp>
#include <dynamic_reconfigure/server.h>
#include <stroll_bearnav/featureExtractionConfig.h>
#include <std_msgs/Int32.h>
#include <sensor_msgs/PointCloud.h>

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
static const std::string OPENCV_WINDOW = "Image window";

/* Detector and descriptor types  */
typedef enum
{
	DET_NONE = 0,
	DET_AGAST,
	DET_SURF,
	DET_UPSURF
} EDetectorType;

typedef enum
{
	DES_NONE = 0,
	DES_BRIEF,
	DES_SURF
} EDescriptorType;

/* Subscribers and publishers  */
image_transport::Subscriber image_sub_;
image_transport::Publisher image_pub_;
ros::Publisher feat_pub_;
ros::Publisher right_points_pub_;
ros::Publisher left_points_pub_;

/* Features  */
stroll_bearnav::FeatureArray featureArray;
stroll_bearnav::Feature feature;

/* Image feature parameters */
float detectionThreshold = 0;

/* Detectors and descriptors */
Ptr<AgastFeatureDetector> agastDetector = AgastFeatureDetector::create(detectionThreshold);
Ptr<BriefDescriptorExtractor> briefDescriptor = BriefDescriptorExtractor::create();
Ptr<SURF> surf = SURF::create(detectionThreshold);
Ptr<SURF> upSurf = SURF::create(detectionThreshold, 4, 3, false, true);

/* Set detector and descriptor */
EDetectorType usedDetector = DET_NONE;
EDescriptorType usedDescriptor = DES_NONE;
NormTypes featureNorm = NORM_INF;

/* Optimization parameter and clock */
bool optimized = false;
clock_t t;

/* Adaptive threshold parameters */
bool adaptThreshold = true;
int targetKeypoints = 1;
float featureOvershootRatio = 0.3;
float maxLine = 0.5;
int target_over;
void adaptive_threshold(vector<KeyPoint> &keypoints);

/* Matcher */
Ptr<DescriptorMatcher> matcher = BFMatcher::create(featureNorm);

/* Vectors for matching */
vector< vector<DMatch> > matches;
vector<DMatch> good_matches;
vector<KeyPoint> left_keypoints, right_keypoints;
vector<float> x_coordinate, y_coordinate, z_coordinate;

/* Mats for matching */
Mat left_descriptors, right_descriptors, img;

NormTypes cFeatureNorm = featureNorm; // ???

/* Ration match constant */
float ratioMatchConstant = 0.3;

/* Count of best matches found per each query descriptor or less if a query descriptor has less than k possible matches in total */
int knn = 5;

/* Constant to find z-coordinate = distance * difference between x */
double c = 0.65 * 0.56; // in m

/* Vertical treshold */
int vertical_threshold = 5;

/* Cloud keypoints */
sensor_msgs::PointCloud right_cloud_keypoints, left_cloud_keypoints;

/* Camera matrix */
Mat cameraMatrix = (Mat_<float>(3,3) << 714.4060659074023, 0.0, 378.35395440554737, 0.0, 714.4060659074023, 211.30912263284839, 0.0, 0.0, 1.0);
//Mat distortionCoefficients = (Mat_<float>(5,1) << 0.07527054569113517, -0.1237814448250655, -0.004382264458101108, -0.0020232366942786123, 0.0);

int detectKeyPoints(Mat &image, vector<KeyPoint> &keypoints)
{
	cv::Mat img;
	if (maxLine < 1.0)
		img = image(cv::Rect(0, 0, image.cols, (int)(image.rows * maxLine)));
	else
		img = image;
	if (usedDetector == DET_AGAST)
		agastDetector->detect(img, keypoints, Mat());
	if (usedDetector == DET_SURF)
		surf->detect(img, keypoints, Mat());
	if (usedDetector == DET_UPSURF)
		upSurf->detect(img, keypoints, Mat());
}

int describeKeyPoints(Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors)
{
	if (usedDescriptor == DES_BRIEF)
		briefDescriptor->compute(img, keypoints, descriptors);
	if (usedDescriptor == DES_SURF)
		surf->compute(img, keypoints, descriptors);
}

int detectAndDescribe(Mat &image, vector<KeyPoint> &keypoints, Mat &descriptors)
{
	if (usedDescriptor == DES_SURF && usedDetector == DET_SURF)
		surf->detectAndCompute(img, Mat(), keypoints, descriptors);
	else if (usedDescriptor == DES_SURF && usedDetector == DET_UPSURF)
		upSurf->detectAndCompute(img, Mat(), keypoints, descriptors);
	else
	{
		detectKeyPoints(image, keypoints);
		describeKeyPoints(image, keypoints, descriptors);
	}
}

int setThreshold(int thres)
{
	if (usedDetector == DET_AGAST)
		agastDetector->setThreshold(thres);
	if (usedDetector == DET_SURF)
		surf->setHessianThreshold(thres);
}

/* Dynamic reconfigure of surf threshold and showing images */
void callback(stroll_bearnav::featureExtractionConfig &config, uint32_t level)
{
	adaptThreshold = config.adaptThreshold;
	if (!adaptThreshold)
		detectionThreshold = config.thresholdParam;
	targetKeypoints = config.targetKeypoints;
	featureOvershootRatio = config.featureOvershootRatio;
	target_over = targetKeypoints + featureOvershootRatio / 100.0 * targetKeypoints;
	maxLine = config.maxLine;

	/* Set ratio match constant, c constant and vertical threshold */
	ratioMatchConstant = config.ratioMatchConstant;
	c = config.cConstant;
	vertical_threshold = config.verticalThreshold;

	/* Optimize detecting features and measure time */
	optimized = config.optimized;
	usedDescriptor = (EDescriptorType)config.descriptor;
	usedDetector = (EDetectorType)config.detector;
	switch (usedDescriptor)
	{
	case DES_BRIEF:
		featureNorm = NORM_HAMMING;
		break;
	case DES_SURF:
		featureNorm = NORM_L2;
		break;
	}
	setThreshold(detectionThreshold);

	ROS_DEBUG("Changing feature featureExtraction to %.3f, keypoints %i", detectionThreshold, targetKeypoints);
}

/* To select most responsive features */
bool compare_response(KeyPoint first, KeyPoint second)
{
	if (first.response > second.response)
		return true;
	else
		return false;
}

/* Extract features from image recieved from camera */
void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
	/* Get image */
	cv_bridge::CvImagePtr cv_ptr;
	try
	{
		cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
	}
	catch (cv_bridge::Exception &e)
	{
		ROS_ERROR("cv_bridge exception: %s", e.what());
		return;
	}
	img = cv_ptr->image;

	/* Clear */
	good_matches.clear();
	left_keypoints.clear();
	right_keypoints.clear();
	matches.clear();
	right_cloud_keypoints.points.clear();
	left_cloud_keypoints.points.clear();
	int number_of_coordinates = 0, number_of_best_matches = 0;
	float average_distance = 0.0;

	/* Split image */
	Mat left_img(img, Rect(0, 0, round(img.cols / 2), img.rows));
	Mat right_img(img, Rect(round(img.cols / 2), 0, round(img.cols / 2), img.rows));

	/* Detect keypoints */
	detectKeyPoints(left_img, left_keypoints);
	sort(left_keypoints.begin(), left_keypoints.end(), compare_response);

	detectKeyPoints(right_img, right_keypoints);
	sort(right_keypoints.begin(), right_keypoints.end(), compare_response);

	/* Determine the next threshold */
	adaptive_threshold(left_keypoints);
	adaptive_threshold(right_keypoints);

	/* Reduce keypoints size to desired number of keypoints */
	left_keypoints.erase(left_keypoints.begin() + min(targetKeypoints, (int)left_keypoints.size()), left_keypoints.end());
	right_keypoints.erase(right_keypoints.begin() + min(targetKeypoints, (int)right_keypoints.size()), right_keypoints.end());

	/* Then compute descriptors only for desired number of keypoints */
	describeKeyPoints(left_img, left_keypoints, left_descriptors);
	describeKeyPoints(right_img, right_keypoints, right_descriptors);

	if (cFeatureNorm != featureNorm)
	{
		matcher = BFMatcher::create(featureNorm);
		cFeatureNorm = featureNorm;
	}

	/* Compare left and right descriptor */
	if (left_keypoints.size() > 0 && right_keypoints.size() > 0)
	{

		/* Feature matching */
		try
		{
			matcher->knnMatch(left_descriptors, right_descriptors, matches, knn);
		}
		catch (Exception &e)
		{
			matches.clear();
			ROS_ERROR("Feature desriptors from the map and in from the image are not compatible.");
		}

		/* Perform ratio matching */
		good_matches.reserve(matches.size());
		left_cloud_keypoints.points.resize(matches.size());
		right_cloud_keypoints.points.resize(matches.size());

		// For all matches
		for (size_t i = 0; i < matches.size(); i++)
		{
			if (matches[i][0].distance < ratioMatchConstant * matches[i][1].distance)
			{
	
				/* Calculate vertical and horizontal differences */
				Point2f right_point = right_keypoints[matches[i][0].trainIdx].pt;
				Point2f left_point = left_keypoints[matches[i][0].queryIdx].pt;
				double difference_ver = fabs(right_point.y - left_point.y);
				double difference_hor = fabs(right_point.x - left_point.x);

				/* Push only straight matches */
				if (difference_ver <= vertical_threshold)
				{
					good_matches.push_back(matches[i][0]); // Add to good matches

					// Calculate z-coordinate
					float cor_z = c / difference_hor; // in m
					z_coordinate.push_back(cor_z);

					// Calculate x, y-coordinate
					Mat point = (Mat_<float>(2,1) << left_point.x, left_point.y), distortionCoefficients, output_points;
  					//undistort(point, output_points, cameraMatrix, distortionCoefficients);
  					//ROS_INFO("X: %f, Y: %f, Z: %f", output_points.at<float>(0, 0), output_points.at<float>(1, 0), output_points.at<float>(2, 0));
					float cor_x = (left_point.x - cameraMatrix.at<float>(0, 2)) / cameraMatrix.at<float>(0, 0) * cor_z;
					//(output_points.at<float>(0, 0) * cor_z; // in m
					float cor_y = (left_point.y - cameraMatrix.at<float>(1, 2)) / cameraMatrix.at<float>(1, 1) * cor_z;
					x_coordinate.push_back(cor_x);
					y_coordinate.push_back(cor_y);
					
					// Show info
					ROS_INFO("RX: %.0f, RY: %.0f", right_point.x, right_point.y);
					ROS_INFO("LX: %.0f, LY: %.0f", left_point.x, left_point.y);
					ROS_INFO("DVER: %.0f, DHOR: %.0f", difference_ver, difference_hor);
					ROS_INFO("X: %f, Y: %f, Z: %f", cor_x, cor_y, cor_z);
					ROS_INFO("-------------------------");

					// Set cloud points (/100 - better to show)
					left_cloud_keypoints.points[i].x = cor_x;
					left_cloud_keypoints.points[i].y = cor_z;
					left_cloud_keypoints.points[i].z = cor_y;

					right_cloud_keypoints.points[i].x = cor_x;
					right_cloud_keypoints.points[i].y = cor_z;
					right_cloud_keypoints.points[i].z = cor_y;
					
					average_distance += cor_z;
					number_of_coordinates++;
				}
				number_of_best_matches++;
			}
		}
	}

	// Set channel depth
	sensor_msgs::ChannelFloat32 depth_channel;
	depth_channel.name = "distance";
	for (int i = 0; i < left_cloud_keypoints.points.size(); i++)
	{
		depth_channel.values.push_back(left_cloud_keypoints.points[i].y);
	}

	// Add channel to point cloud
	left_cloud_keypoints.channels.push_back(depth_channel);
	right_cloud_keypoints.channels.push_back(depth_channel);

	// Set header of cloud keypoints
	left_cloud_keypoints.header.frame_id = "map";
	left_cloud_keypoints.header.stamp = ros::Time::now();

	right_cloud_keypoints.header.frame_id = "map";
	right_cloud_keypoints.header.stamp = ros::Time::now();

	// Publish right and left cloud keypoints
	left_points_pub_.publish(left_cloud_keypoints);
	right_points_pub_.publish(right_cloud_keypoints);

	/* Publish image features */
	featureArray.feature.clear();
	for (int i = 0; i < left_keypoints.size(); i++)
	{ // Only left keypoints
		feature.x = left_keypoints[i].pt.x;
		feature.y = left_keypoints[i].pt.y;
		feature.cor_x = x_coordinate[i];
		feature.cor_y = y_coordinate[i];
		feature.cor_z = z_coordinate[i];
		feature.size = left_keypoints[i].size;
		feature.angle = left_keypoints[i].angle;
		feature.response = left_keypoints[i].response;
		feature.octave = left_keypoints[i].octave;
		feature.class_id = featureNorm;

		left_descriptors.row(i).copyTo(feature.descriptor);

		if (adaptThreshold)
		{
			if (i < targetKeypoints)
				featureArray.feature.push_back(feature);
		}
		else
		{
			featureArray.feature.push_back(feature);
		}
	}

	char numStr[100];
	sprintf(numStr, "Image_%09d", msg->header.seq);
	featureArray.id = numStr;

	featureArray.distance = msg->header.seq;
	printf("Features: %i, Matches: %i, BestMatches: %i, Coordinates: %i, Average distance: %f\n", (int)featureArray.feature.size(), (int)matches.size(), number_of_best_matches, number_of_coordinates, average_distance/number_of_coordinates);
	feat_pub_.publish(featureArray);

	/* Show image with good matches */
	if (left_keypoints.size() > 0 && right_keypoints.size() > 0 && image_pub_.getNumSubscribers() > 0)
	{
		/* Draw keypoints and matches */
		Mat output;
		drawMatches(left_img, left_keypoints, right_img, right_keypoints, good_matches, output, Scalar(0, 0, 255), Scalar(0, 0, 255), vector<char>(), 0);

		/* Send image */
		std_msgs::Header header;
		cv_bridge::CvImage bridge(header, sensor_msgs::image_encodings::BGR8, output);
		image_pub_.publish(bridge.toImageMsg());
	}
}

/* Adaptive threshold - trying to target a given number of keypoints */
void adaptive_threshold(vector<KeyPoint> &keypoints)
{
	// Supposes keypoints are sorted according to response (applies to surf)
	if (keypoints.size() > target_over)
	{
		detectionThreshold = ((keypoints[target_over].response + keypoints[target_over + 1].response) / 2);
		//ROS_INFO("Keypoints %ld over  %i, missing %4ld, set threshold %.3f between responses %.3f %.3f",keypoints.size(),target_over, target_over - keypoints.size(),detectionThreshold,keypoints[target_over].response,keypoints[target_over + 1].response);
	}
	else
	{
		/* Compute average difference between responses of n last keypoints */
		if (keypoints.size() > 7)
		{
			int n_last = (int)round(keypoints.size() / 5);
			float avg_dif = 0;

			for (int j = (keypoints.size() - n_last); j < keypoints.size() - 1; ++j)
			{
				avg_dif += keypoints[j].response - keypoints[j + 1].response;
			}

			detectionThreshold -= avg_dif / (n_last - 1) * (target_over - keypoints.size());
			//ROS_INFO("Keypoints %ld under %i, missing %4ld, set threshold %.3f from %i last features with %.3f difference",keypoints.size(),target_over,target_over - keypoints.size(),detectionThreshold,n_last,avg_dif);
		}
		else
		{
			detectionThreshold = 0;
			//ROS_INFO("Keypoints %ld under %i, missing %4ld, set threshold %.3f ",keypoints.size(),target_over,target_over - keypoints.size(),detectionThreshold);
		}
	}
	detectionThreshold = fmax(detectionThreshold, 0);
	setThreshold(detectionThreshold);
}

void keypointCallback(const std_msgs::Int32::ConstPtr &msg)
{
	targetKeypoints = msg->data;
	target_over = targetKeypoints + featureOvershootRatio / 100.0 * targetKeypoints;
	//ROS_INFO("targetKeypoints set to %i, overshoot: %i",targetKeypoints,target_over);
}

int main(int argc, char **argv)
{
	ros::init(argc, argv, "feature_extraction");
	ros::NodeHandle nh_;
	image_transport::ImageTransport it_(nh_);

	/* Initiate dynamic reconfiguration */
	dynamic_reconfigure::Server<stroll_bearnav::featureExtractionConfig> server;
	dynamic_reconfigure::Server<stroll_bearnav::featureExtractionConfig>::CallbackType f = boost::bind(&callback, _1, _2);
	server.setCallback(f);

	// Pubs
	feat_pub_ = nh_.advertise<stroll_bearnav::FeatureArray>("/features", 1);
	left_points_pub_ = nh_.advertise<sensor_msgs::PointCloud>("/pointCloud/left", 10);
	right_points_pub_ = nh_.advertise<sensor_msgs::PointCloud>("/pointCloud/right", 10);
	image_pub_ = it_.advertise("/image_with_features", 1);

	// Subs
	image_sub_ = it_.subscribe("/image", 1, imageCallback);
	ros::Subscriber key_sub = nh_.subscribe("/targetKeypoints", 1, keypointCallback);

	ros::spin();
	return 0;
}
