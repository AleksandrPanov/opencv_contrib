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

}