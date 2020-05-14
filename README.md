# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* Git LFS
  * Weight files are handled using [LFS](https://git-lfs.github.com/)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.

## Tasks

FP.1 Match 3D Objects

```
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
```

FP.2 Compute Lidar-based TTC

```
void computeTTCLidar(std::vector<LidarPoint>& lidarPointsPrev,
    std::vector<LidarPoint>& lidarPointsCurr, double frameRate, double& TTC)
{
    //Only the points within the ego line 2 meters will be considered
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
    // avoid divede by 0
    double temp = std::fabs(distPrev - distCurr) > 0 ? distPrev - distCurr: 0.0001;

    TTC = distCurr / (temp * frameRate);
}
```

FP.3 Associate Keypoint Correspondences with Bounding Boxes

```
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    for (cv::DMatch match : kptMatches) {
        // if the point from previous and current frame if within the current bounding box (of course this if not accurate enough, if more accurate is required, the bounding box from previous frames are needed)
        if (boundingBox.roi.contains(kptsCurr[match.trainIdx].pt) && boundingBox.roi.contains(kptsPrev[match.queryIdx].pt))
            boundingBox.kptMatches.emplace_back(match);
    }
}
```

FP.4 Compute Camera-based TTC

```
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    std::vector<double> distRatios;
    cv::KeyPoint matchedPointPrev1, matchedPointCurr1, matchedPointPrev2, matchedPointCurr2;
    double ratio;
    // loop through all the matches
    for (auto match = kptMatches.begin(); match != kptMatches.end(); match++) {
        matchedPointCurr1 = kptsCurr[match->trainIdx];
        matchedPointPrev1 = kptsPrev[match->queryIdx];
        // make sure every combination of mathes only be ran once
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
    }
    if (distRatios.size() % 2 == 0) {
        TTC = - 1 / (frameRate * (1 - distRatios[distRatios.size() / 2]));
    }
    else {
        TTC = - 1 / (frameRate * (1 - distRatios[(distRatios.size() - 1) / 2]));
    }
}

```

FP.5 Performance Evaluation 1

TTC based on lidar seems sometimes goes wrong when some outliers are included, such as the points that go through the front mirror of the preceding vehicles.
Keep the shrinking factor to a relatively bigger number (e.g 0.2) can avoid this happen. 
Also, some machine learning methods can be applied, for example, clustering, to exclude outliers.

FP.6 Performance Evaluation 2

Exploring all the possible combinations of detector and descriptor (AKAZE descriptor only works with AKAZE_Point, SIFT doesn't work with ORB), the results are concluded in the following.

* Detector SHITOMASI + Descriptor BRISK in total 58.448 seconds; L1 norm of TTC between camera and radar 1.16692; False camera detection frames: 0
* Detector SHITOMASI + Descriptor BRIEF in total 40.0382 seconds; L1 norm of TTC between camera and radar 1.57872; False camera detection frames: 0
* Detector SHITOMASI + Descriptor ORB in total 25.4711 seconds; L1 norm of TTC between camera and radar 1.6841; False camera detection frames: 0
* Detector SHITOMASI + Descriptor FREAK in total 27.6701 seconds; L1 norm of TTC between camera and radar 1.41911; False camera detection frames: 0
* Detector SHITOMASI + Descriptor SIFT in total 22.3277 seconds; L1 norm of TTC between camera and radar 1.42208; False camera detection frames: 0
* Detector HARRIS + Descriptor BRISK in total 26.399 seconds; L1 norm of TTC between camera and radar 24.5881; False camera detection frames: 6
* Detector HARRIS + Descriptor BRIEF in total 20.9176 seconds; L1 norm of TTC between camera and radar 6.64543; False camera detection frames: 4
* Detector HARRIS + Descriptor ORB in total 21.3883 seconds; L1 norm of TTC between camera and radar 4.66613; False camera detection frames: 5
* Detector HARRIS + Descriptor FREAK in total 23.3505 seconds; L1 norm of TTC between camera and radar 3.61567; False camera detection frames: 5
* Detector HARRIS + Descriptor SIFT in total 23.2715 seconds; L1 norm of TTC between camera and radar 6.90552; False camera detection frames: 4
* Detector FAST + Descriptor BRISK in total 35.4268 seconds; L1 norm of TTC between camera and radar 1.65707; False camera detection frames: 0
* Detector FAST + Descriptor BRIEF in total 28.0267 seconds; L1 norm of TTC between camera and radar 1.51435; False camera detection frames: 0
* Detector FAST + Descriptor ORB in total 28.0081 seconds; L1 norm of TTC between camera and radar 1.47106; False camera detection frames: 0
* Detector FAST + Descriptor FREAK in total 30.0154 seconds; L1 norm of TTC between camera and radar 1.38915; False camera detection frames: 0
* Detector FAST + Descriptor SIFT in total 37.1256 seconds; L1 norm of TTC between camera and radar 1.44513; False camera detection frames: 0
* Detector BRISK + Descriptor BRISK in total 40.3832 seconds; L1 norm of TTC between camera and radar 3.21155; False camera detection frames: 0
* Detector BRISK + Descriptor BRIEF in total 33.0887 seconds; L1 norm of TTC between camera and radar 2.86033; False camera detection frames: 0
* Detector BRISK + Descriptor ORB in total 33.0039 seconds; L1 norm of TTC between camera and radar 2.81707; False camera detection frames: 0
* Detector BRISK + Descriptor FREAK in total 33.6697 seconds; L1 norm of TTC between camera and radar 2.95996; False camera detection frames: 0
* Detector BRISK + Descriptor SIFT in total 39.3464 seconds; L1 norm of TTC between camera and radar 2.85227; False camera detection frames: 0
* Detector ORB + Descriptor BRISK in total 30.8544 seconds; L1 norm of TTC between camera and radar 10.0837; False camera detection frames: 1
* Detector ORB + Descriptor BRIEF in total 24.2895 seconds; L1 norm of TTC between camera and radar 18.7086; False camera detection frames: 0
* Detector ORB + Descriptor ORB in total 24.3139 seconds; L1 norm of TTC between camera and radar 49.0748; False camera detection frames: 1
* Detector ORB + Descriptor FREAK in total 25.2428 seconds; L1 norm of TTC between camera and radar 29.4407; False camera detection frames: 2
* Detector ORB + Descriptor SIFT in total 26.8022 seconds; L1 norm of TTC between camera and radar 28.7031; False camera detection frames: 2
* Detector AKAZE + Descriptor BRISK in total 32.5661 seconds; L1 norm of TTC between camera and radar 1.14068; False camera detection frames: 0
* Detector AKAZE + Descriptor BRIEF in total 26.1364 seconds; L1 norm of TTC between camera and radar 1.06513; False camera detection frames: 0
* Detector AKAZE + Descriptor ORB in total 25.9498 seconds; L1 norm of TTC between camera and radar 1.11386; False camera detection frames: 0
* Detector AKAZE + Descriptor FREAK in total 26.8674 seconds; L1 norm of TTC between camera and radar 0.985288; False camera detection frames: 0
* Detector AKAZE + Descriptor AKAZE in total 27.3121 seconds; L1 norm of TTC between camera and radar 1.15949; False camera detection frames: 0
* Detector AKAZE + Descriptor SIFT in total 27.2958 seconds; L1 norm of TTC between camera and radar 1.18159; False camera detection frames: 0
* Detector SIFT + Descriptor BRISK in total 33.9686 seconds; L1 norm of TTC between camera and radar 1.24448; False camera detection frames: 0
* Detector SIFT + Descriptor BRIEF in total 27.485 seconds; L1 norm of TTC between camera and radar 1.22792; False camera detection frames: 0
* Detector SIFT + Descriptor FREAK in total 28.0052 seconds; L1 norm of TTC between camera and radar 1.25243; False camera detection frames: 0
* Detector SIFT + Descriptor SIFT in total 30.2348 seconds; L1 norm of TTC between camera and radar 1.22499; False camera detection frames: 0


Considering the processing speed, the winners are HARRIS+ORB, SHITOMASI+SIFT, HARRIS+BREIF.
Considering the error (false estimation frames excluded), the winners are AKAZA+FREAK, AKAZA+BRIEF, AKAZA+BRISK.
