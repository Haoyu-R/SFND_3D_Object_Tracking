
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <set>
#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0);
        
        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes
        
        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    //cv::resize(topviewImg, topviewImg, cv::Size(200, 200));
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for (cv::DMatch match : kptMatches) {
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt) && boundingBox.roi.contains(kptsPrev[match.queryIdx].pt))
            boundingBox.kptMatches.emplace_back(match);
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    std::vector<double> distRatios;
    cv::KeyPoint matchedPointPrev1, matchedPointCurr1, matchedPointPrev2, matchedPointCurr2;
    double ratio;

    for (auto match = kptMatches.begin(); match != kptMatches.end(); match++) {
        matchedPointCurr1 = kptsCurr[match->trainIdx];
        matchedPointPrev1 = kptsPrev[match->queryIdx];
        for (auto succeedMatch = match+1; succeedMatch != kptMatches.end(); succeedMatch++) {
            matchedPointCurr2 = kptsCurr[succeedMatch->trainIdx];
            matchedPointPrev2 = kptsPrev[succeedMatch->queryIdx];

            double distCurr = cv::norm(matchedPointCurr1.pt - matchedPointCurr2.pt);
            double distPrev = cv::norm(matchedPointPrev1.pt - matchedPointPrev2.pt);
           
            // Avoid outlier and divide by 0
            if(distCurr > 100 && distPrev > std::numeric_limits<double>::epsilon())
                distRatios.emplace_back(distCurr/distPrev);
        }
    }
    
    std::sort(distRatios.begin(), distRatios.end());

    if (distRatios.size() < 1) {
        TTC = NAN;
        return;
    }
    if (distRatios.size() % 2 == 0) {
        TTC = - 1 / (frameRate * (1 - distRatios[distRatios.size() / 2]));
    }
    else {
        TTC = - 1 / (frameRate * (1 - distRatios[(distRatios.size() - 1) / 2]));
    }
}


void computeTTCLidar(std::vector<LidarPoint>& lidarPointsPrev,
    std::vector<LidarPoint>& lidarPointsCurr, double frameRate, double& TTC)
{
    //Only the points within the ego line will be considered
    double distPrev = 0, distCurr = 0;
    int countPrev = 0, countCurr = 0;
    for (auto point : lidarPointsCurr) {
        if (fabs(point.y) < 2) {
            distCurr += point.x;
            countCurr++;
        }
    }
    for (auto point : lidarPointsPrev) {
        if (fabs(point.y) < 2) {
            distPrev += point.x;
            countPrev++;
        }
    }
    
    distPrev /= countPrev;
    distCurr /= countCurr;
    double temp = std::fabs(distPrev - distCurr) > 0 ? distPrev - distCurr: 0.0001;

    TTC = distCurr / (temp * frameRate);
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    std::set<int> keys;

    std::multimap<int, int> matchBoxes;

    // use multimap to store all the boxID associated to every key points in prev and curr frame

    for (auto i = matches.begin(); i != matches.end(); i++) {

        int prev, curr;

        bool foundInPrev = false, foundInCurr = false;
        int prevBoxID, currBoxID;

        for (auto boxIt = prevFrame.boundingBoxes.begin(); boxIt != prevFrame.boundingBoxes.end(); boxIt++) {
            if (boxIt->roi.contains(prevFrame.keypoints[i->queryIdx].pt)) {
                prev = boxIt->boxID;
                foundInPrev = true;
                break;
            }
        }

        for (auto boxIt = currFrame.boundingBoxes.begin(); boxIt != currFrame.boundingBoxes.end(); boxIt++) {
            if (boxIt->roi.contains(currFrame.keypoints[i->trainIdx].pt)) {
                curr = boxIt->boxID;
                foundInCurr = true;
                break;
            }
        }

        if (foundInPrev && foundInCurr) {
            keys.insert(prev);
            matchBoxes.insert({ prev, curr });
        }
     
    }
    // Didn't find anything then return
    if (keys.size() < 1)
        return;

    // for every box in prev frame search the best match in curr fram
    for (auto it = keys.begin(); it != keys.end(); it++) {
        typedef std::multimap<int, int>::iterator MMAPIterator;
        // find all the boxex in the curr frame associated with each box in last frame 
        std::pair<MMAPIterator, MMAPIterator> result = matchBoxes.equal_range(*it);

        // use map to find most occurency/matches of one box in prev frame with curr frame;
        std::map<int, size_t> frequencyCount;

        for (MMAPIterator it2 = result.first; it2 != result.second; it2++) {
            frequencyCount[it2->second]++;
        }

        auto pr = std::max_element(
            frequencyCount.begin(), frequencyCount.end(), [](auto i, auto j) {return i.second < j.second; }
        );
        // Insert
        bbBestMatches.insert({*it, pr->first});
    
    }
}
