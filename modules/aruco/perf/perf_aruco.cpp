// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

namespace opencv_test {
using namespace perf;

// useAruco3Detection
// CORNER_REFINE_SUBPIX
typedef tuple <bool, int> ArucoTestParams;
typedef TestBaseWithParam<ArucoTestParams> EstimateAruco;
#define ESTIMATE_PARAMS Combine(Values(false, true), Values(-1))

PERF_TEST_P(EstimateAruco, ArucoFirst, ESTIMATE_PARAMS )
{
    string imgPath = cvtest::findDataFile("gboriginal.png", false);
    Mat image = imread(imgPath);
    string dictPath = cvtest::findDataFile("tutorial_dict.yml", false);
    cv::Ptr<cv::aruco::Dictionary> dictionary;

    FileStorage fs(dictPath, FileStorage::READ);
    if (!aruco::Dictionary::readDictionary(fs.root(), dictionary)) {
        cvtest::SkipTestException("Not founded tutorial_dict.yml");
    }

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    ArucoTestParams params = GetParam();
    detectorParams->useAruco3Detection = get<0>(params);
    detectorParams->minMarkerLengthRatioOriginalImg = 0.01f;

    vector<int> ids;
    vector<vector<Point2f> > corners, rejected;
    const size_t N = 35ull;
    // corners of ArUco markers with indices 0, 1, ..., 34
    const int goldCorners[N][8] = { {252,74, 286,81, 274,102, 238,95},    {295,82, 330,89, 319,111, 282,104},
                                    {338,91, 375,99, 365,121, 327,113},   {383,100, 421,107, 412,130, 374,123},
                                    {429,109, 468,116, 461,139, 421,132}, {235,100, 270,108, 257,130, 220,122},
                                    {279,109, 316,117, 304,140, 266,133}, {324,119, 362,126, 352,150, 313,143},
                                    {371,128, 410,136, 400,161, 360,152}, {418,139, 459,145, 451,170, 410,163},
                                    {216,128, 253,136, 239,161, 200,152}, {262,138, 300,146, 287,172, 248,164},
                                    {309,148, 349,156, 337,183, 296,174}, {358,158, 398,167, 388,194, 346,185},
                                    {407,169, 449,176, 440,205, 397,196}, {196,158, 235,168, 218,195, 179,185},
                                    {243,170, 283,178, 269,206, 228,197}, {293,180, 334,190, 321,218, 279,209},
                                    {343,192, 385,200, 374,230, 330,220}, {395,203, 438,211, 429,241, 384,233},
                                    {174,192, 215,201, 197,231, 156,221}, {223,204, 265,213, 249,244, 207,234},
                                    {275,215, 317,225, 303,257, 259,246}, {327,227, 371,238, 359,270, 313,259},
                                    {381,240, 426,249, 416,282, 369,273}, {151,228, 193,238, 173,271, 130,260},
                                    {202,241, 245,251, 228,285, 183,274}, {255,254, 300,264, 284,299, 238,288},
                                    {310,267, 355,278, 342,314, 295,302}, {366,281, 413,290, 402,327, 353,317},
                                    {125,267, 168,278, 147,314, 102,303}, {178,281, 223,293, 204,330, 157,317},
                                    {233,296, 280,307, 263,346, 214,333}, {291,310, 338,322, 323,363, 274,349},
                                    {349,325, 399,336, 386,378, 335,366} };
    map<int, const int*> mapGoldCorners;
    for (int i = 0; i < static_cast<int>(N); i++)
        mapGoldCorners[i] = goldCorners[i];

    TEST_CYCLE() aruco::detectMarkers(image, dictionary, corners, ids, detectorParams, rejected);
    SANITY_CHECK_NOTHING();
}

static double deg2rad(double deg) { return deg * CV_PI / 180.; }

class MarkerPainter
{
private:
    Size imgMarkerSize;
    Mat cameraMatrix;
public:
    MarkerPainter(const Size& size)
    {
        setImgMarkerSize(size);
    }

    void setImgMarkerSize(const Size& size)
    {
        imgMarkerSize = size;
        CV_Assert(imgMarkerSize.width == imgMarkerSize.height);
        cameraMatrix = Mat::eye(3, 3, CV_64FC1);
        cameraMatrix.at<double>(0, 0) = cameraMatrix.at<double>(1, 1) = 650;
        cameraMatrix.at<double>(0, 2) = imgMarkerSize.width / 2;
        cameraMatrix.at<double>(1, 2) = imgMarkerSize.height / 2;
    }

    static std::pair<Mat, Mat> getSyntheticRT(double yaw, double pitch, double distance)
    {
        auto rvec_tvec = std::make_pair(Mat(3, 1, CV_64FC1), Mat(3, 1, CV_64FC1));
        Mat& rvec = rvec_tvec.first;
        Mat& tvec = rvec_tvec.second;

        // Rvec
        // first put the Z axis aiming to -X (like the camera axis system)
        Mat rotZ(3, 1, CV_64FC1);
        rotZ.ptr<double>(0)[0] = 0;
        rotZ.ptr<double>(0)[1] = 0;
        rotZ.ptr<double>(0)[2] = -0.5 * CV_PI;

        Mat rotX(3, 1, CV_64FC1);
        rotX.ptr<double>(0)[0] = 0.5 * CV_PI;
        rotX.ptr<double>(0)[1] = 0;
        rotX.ptr<double>(0)[2] = 0;

        Mat camRvec, camTvec;
        composeRT(rotZ, Mat(3, 1, CV_64FC1, Scalar::all(0)), rotX, Mat(3, 1, CV_64FC1, Scalar::all(0)),
                  camRvec, camTvec);

        // now pitch and yaw angles
        Mat rotPitch(3, 1, CV_64FC1);
        rotPitch.ptr<double>(0)[0] = 0;
        rotPitch.ptr<double>(0)[1] = pitch;
        rotPitch.ptr<double>(0)[2] = 0;

        Mat rotYaw(3, 1, CV_64FC1);
        rotYaw.ptr<double>(0)[0] = yaw;
        rotYaw.ptr<double>(0)[1] = 0;
        rotYaw.ptr<double>(0)[2] = 0;

        composeRT(rotPitch, Mat(3, 1, CV_64FC1, Scalar::all(0)), rotYaw,
                  Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec, tvec);

        // compose both rotations
        composeRT(camRvec, Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec,
                  Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec, tvec);

        // Tvec, just move in z (camera) direction the specific distance
        tvec.ptr<double>(0)[0] = 0.;
        tvec.ptr<double>(0)[1] = 0.;
        tvec.ptr<double>(0)[2] = distance;
        return rvec_tvec;
}

    std::pair<Mat, vector<Point2f> > getProjectMarker(int id, double yaw, double pitch,
                                                      const Ptr<aruco::DetectorParameters>& parameters,
                                                      const Ptr<aruco::Dictionary>& dictionary)
    {
        CV_Assert(imgMarkerSize.width == imgMarkerSize.height);
        auto marker_corners = std::make_pair(Mat(imgMarkerSize, CV_8UC1, Scalar::all(255)), vector<Point2f>());
        Mat& img = marker_corners.first;
        vector<Point2f>& corners = marker_corners.second;

        // canonical image
        const int markerSizePixels = static_cast<int>(imgMarkerSize.width/sqrt(2));
        aruco::drawMarker(dictionary, id, markerSizePixels, img, parameters->markerBorderBits);

        // get rvec and tvec for the perspective
        const double distance = 0.4;
        auto rvec_tvec = MarkerPainter::getSyntheticRT(yaw, pitch, distance);
        Mat& rvec = rvec_tvec.first;
        Mat& tvec = rvec_tvec.second;

        const float markerLength = 0.05f;
        vector<Point3f> markerObjPoints;
        markerObjPoints.push_back(Point3f(-markerLength / 2.f, +markerLength / 2.f, 0));
        markerObjPoints.push_back(markerObjPoints[0] + Point3f(markerLength, 0, 0));
        markerObjPoints.push_back(markerObjPoints[0] + Point3f(markerLength, -markerLength, 0));
        markerObjPoints.push_back(markerObjPoints[0] + Point3f(0, -markerLength, 0));

        // project markers and draw them
        Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
        projectPoints(markerObjPoints, rvec, tvec, cameraMatrix, distCoeffs, corners);

        vector<Point2f> originalCorners;
        originalCorners.push_back(Point2f(0.f, 0.f));
        originalCorners.push_back(originalCorners[0]+Point2f((float)markerSizePixels, 0));
        originalCorners.push_back(originalCorners[0]+Point2f((float)markerSizePixels, (float)markerSizePixels));
        originalCorners.push_back(originalCorners[0]+Point2f(0, (float)markerSizePixels));

        Mat transformation = getPerspectiveTransform(originalCorners, corners);

        warpPerspective(img, img, transformation, imgMarkerSize, INTER_NEAREST, BORDER_CONSTANT,
                        Scalar::all(255));
        return marker_corners;
    }

    std::pair<Mat, map<int, vector<Point2f> > > getProjectMarkersTile(const int numMarkers,
                                                           const Ptr<aruco::DetectorParameters>& params,
                                                           const Ptr<aruco::Dictionary>& dictionary)
    {
        params->minDistanceToBorder = 1;
        //USE_ARUCO3
        params->useAruco3Detection = true;
        params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        params->minSideLengthCanonicalImg = 32;
        params->markerBorderBits = 1;
        //Mat tileImage(imgMarkerSize*numMarkers, CV_8UC1, Scalar::all(255));
        Mat tileImage(Size(240, 240), CV_8UC1, Scalar::all(255));
        map<int, vector<Point2f> > idCorners;

        int iter = 0, pitch = 0, yaw = 70;
        for (int i = 0; i < numMarkers; i++)
        {
            for (int j = 0; j < numMarkers; j++)
            {
                int currentId = iter % 250;
                auto marker_corners = getProjectMarker(currentId, deg2rad(yaw), deg2rad(pitch), params, dictionary);
                Mat tmp_roi = tileImage(Rect(i*imgMarkerSize.height, j*imgMarkerSize.width, imgMarkerSize.height, imgMarkerSize.width));
                marker_corners.first.copyTo(tmp_roi);

                for (Point2f& point: marker_corners.second)
                    point += Point2f(i*i*imgMarkerSize.height, j*i*imgMarkerSize.width);
                idCorners[currentId] = marker_corners.second;
                yaw = (yaw + 40) % 120;
                currentId++;
            }
            pitch = (pitch + 70) % 360;
        }
        imwrite("tile_test" + std::to_string(iter) + ".jpg", tileImage);
        return std::make_pair(std::move(tileImage), std::move(idCorners));
    }
};

TEST(EstimateAruco, ArucoThird)
{
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);
    Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
    params->minDistanceToBorder = 1;

    //USE_ARUCO3
    params->useAruco3Detection = true;
    params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    params->minSideLengthCanonicalImg = 32;
    params->markerBorderBits = 1;
    MarkerPainter painter(Size(240, 240));
    //painter.getProjectMarkersTile(2, params, dictionary);
    int iter = 0;
    // detect from different positions
    for(int pitch = 0; pitch < 360; pitch += 70) {
        for (int yaw = 70; yaw <= 120; yaw += 40) {
            int currentId = iter % 250;
            auto marker_corners = painter.getProjectMarker(currentId, deg2rad(yaw), deg2rad(pitch), params, dictionary);
            //rectangle(img, groundTruthCorners[0]/4, 3*groundTruthCorners[0]/4, Scalar(0, 0, 0), 4);
            //rectangle(img, Point2i(img.rows, img.cols)*0.9, Point2i(img.rows, img.cols)*0.95, Scalar(0, 0, 0), 4);
            //imwrite("test" + std::to_string(iter) + ".jpg", marker_corners.first);

            // detect markers
            vector<vector<Point2f> > corners;
            vector<int> ids;
            aruco::detectMarkers(marker_corners.first, dictionary, corners, ids, params);
            // check results
            ASSERT_EQ(1ull, ids.size());
            ASSERT_EQ(currentId, ids[0]);
            for(int c = 0; c < 4; c++)
            {
                double dist = cv::norm(marker_corners.second[c] - corners[0][c]);
                EXPECT_LE(dist, 5.0);
            }
            iter++;
        }
    }
}

}