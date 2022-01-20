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
    aruco::Dictionary::readDictionary(fs.root(), dictionary); // set marker from tutorial_dict.yml

    Ptr<aruco::DetectorParameters> detectorParams = aruco::DetectorParameters::create();
    ArucoTestParams params = GetParam();
    detectorParams->useAruco3Detection = get<0>(params);
    detectorParams->minMarkerLengthRatioOriginalImg = 0.01f;

    vector< int > ids;
    vector< vector< Point2f > > corners, rejected;
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

/**
 * @brief Get rvec and tvec from yaw, pitch and distance
 */
static void getSyntheticRT(double yaw, double pitch, double distance, Mat &rvec, Mat &tvec) {

    rvec = Mat(3, 1, CV_64FC1);
    tvec = Mat(3, 1, CV_64FC1);

    // Rvec
    // first put the Z axis aiming to -X (like the camera axis system)
    Mat rotZ(3, 1, CV_64FC1);
    rotZ.ptr< double >(0)[0] = 0;
    rotZ.ptr< double >(0)[1] = 0;
    rotZ.ptr< double >(0)[2] = -0.5 * CV_PI;

    Mat rotX(3, 1, CV_64FC1);
    rotX.ptr< double >(0)[0] = 0.5 * CV_PI;
    rotX.ptr< double >(0)[1] = 0;
    rotX.ptr< double >(0)[2] = 0;

    Mat camRvec, camTvec;
    composeRT(rotZ, Mat(3, 1, CV_64FC1, Scalar::all(0)), rotX, Mat(3, 1, CV_64FC1, Scalar::all(0)),
              camRvec, camTvec);

    // now pitch and yaw angles
    Mat rotPitch(3, 1, CV_64FC1);
    rotPitch.ptr< double >(0)[0] = 0;
    rotPitch.ptr< double >(0)[1] = pitch;
    rotPitch.ptr< double >(0)[2] = 0;

    Mat rotYaw(3, 1, CV_64FC1);
    rotYaw.ptr< double >(0)[0] = yaw;
    rotYaw.ptr< double >(0)[1] = 0;
    rotYaw.ptr< double >(0)[2] = 0;

    composeRT(rotPitch, Mat(3, 1, CV_64FC1, Scalar::all(0)), rotYaw,
              Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec, tvec);

    // compose both rotations
    composeRT(camRvec, Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec,
              Mat(3, 1, CV_64FC1, Scalar::all(0)), rvec, tvec);

    // Tvec, just move in z (camera) direction the specific distance
    tvec.ptr< double >(0)[0] = 0.;
    tvec.ptr< double >(0)[1] = 0.;
    tvec.ptr< double >(0)[2] = distance;
}

/**
 * @brief Create a synthetic image of a marker with perspective
 */
static Mat projectMarker(Ptr<aruco::Dictionary> &dictionary, int id, Mat cameraMatrix, double yaw,
                         double pitch, double distance, Size imageSize, int markerBorder,
                         vector< Point2f > &corners) {
    // canonical image
    Mat marker, markerImg;
    const int markerSizePixels = 100;

    aruco::drawMarker(dictionary, id, markerSizePixels, marker, markerBorder);
    marker.copyTo(markerImg);

    // get rvec and tvec for the perspective
    Mat rvec, tvec;
    getSyntheticRT(yaw, pitch, distance, rvec, tvec);

    const float markerLength = 0.05f;
    vector< Point3f > markerObjPoints;
    markerObjPoints.push_back(Point3f(-markerLength / 2.f, +markerLength / 2.f, 0));
    markerObjPoints.push_back(markerObjPoints[0] + Point3f(markerLength, 0, 0));
    markerObjPoints.push_back(markerObjPoints[0] + Point3f(markerLength, -markerLength, 0));
    markerObjPoints.push_back(markerObjPoints[0] + Point3f(0, -markerLength, 0));

    // project markers and draw them
    Mat distCoeffs(5, 1, CV_64FC1, Scalar::all(0));
    projectPoints(markerObjPoints, rvec, tvec, cameraMatrix, distCoeffs, corners);

    vector< Point2f > originalCorners;
    originalCorners.push_back(Point2f(0.f, 0.f));
    originalCorners.push_back(originalCorners[0]+Point2f((float)markerSizePixels, 0));
    originalCorners.push_back(originalCorners[0]+Point2f((float)markerSizePixels, (float)markerSizePixels));
    originalCorners.push_back(originalCorners[0]+Point2f(0, (float)markerSizePixels));

    Mat transformation = getPerspectiveTransform(originalCorners, corners);

    Mat img(imageSize, CV_8UC1, Scalar::all(255));
    Mat aux;
    const char borderValue = 127;
    warpPerspective(markerImg, aux, transformation, imageSize, INTER_NEAREST, BORDER_CONSTANT,
                    Scalar::all(borderValue));

    // copy only not-border pixels
    for(int y = 0; y < aux.rows; y++) {
        for(int x = 0; x < aux.cols; x++) {
            if(aux.at< unsigned char >(y, x) == borderValue) continue;
            img.at< unsigned char >(y, x) = aux.at< unsigned char >(y, x);
        }
    }

    return img;
}

TEST(EstimateAruco, ArucoSecond) {
    Mat cameraMatrix = Mat::eye(3, 3, CV_64FC1);
    Size imgSize(500, 500); // == 3840 x 2160 pixels == 4K
    //Size imgSize(2880, 2880); // == 3840 x 2160 pixels == 4K
    cameraMatrix.at< double >(0, 0) = cameraMatrix.at< double >(1, 1) = 650;
    cameraMatrix.at< double >(0, 2) = imgSize.width / 2;
    cameraMatrix.at< double >(1, 2) = imgSize.height / 2;
    Ptr<aruco::Dictionary> dictionary = aruco::getPredefinedDictionary(aruco::DICT_6X6_250);

    Ptr<aruco::DetectorParameters> params = aruco::DetectorParameters::create();
    params->minDistanceToBorder = 1;
    // marker :: Inverted
    //CV_ArucoDetectionPerspective::DETECT_INVERTED_MARKER
    //img = ~img;
    //params->detectInvertedMarker = true;
    // CV_ArucoDetectionPerspective::USE_APRILTAG
    //params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_APRILTAG;
    //CV_ArucoDetectionPerspective::USE_ARUCO3
    params->useAruco3Detection = true;
    params->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
    params->minMarkerLengthRatioOriginalImg = 0.05;
    int iter = 0;
    int markerBorder = 1;//iter % 2 + 1;
    params->markerBorderBits = markerBorder;
    // detect from different positions
    for(double distance = 0.1; distance < 0.7; distance += 1.2) {
        for(int pitch = 0; pitch < 360; pitch += 70) {
            for(int yaw = 70; yaw <= 120; yaw += 40){
                int currentId = iter % 250;
                iter++;
                vector< Point2f > groundTruthCorners;

                /// create synthetic image
                Mat img=
                    projectMarker(dictionary, currentId, cameraMatrix, deg2rad(yaw), deg2rad(pitch),
                                  distance, imgSize, markerBorder, groundTruthCorners);
                imwrite("test" + std::to_string(iter) + ".jpg", img);

                // detect markers
                vector< vector< Point2f > > corners;
                vector< int > ids;
                aruco::detectMarkers(img, dictionary, corners, ids, params);

                // check results
                ASSERT_EQ(1ull, ids.size());
                ASSERT_EQ(currentId, ids[0]);

                for(int c = 0; c < 4; c++) {
                    double dist = cv::norm(groundTruthCorners[c] - corners[0][c]);  // TODO cvtest
                    EXPECT_LE(dist, 5.0);
                }
            }
        }
    }
}

}