// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

#include <vector>

namespace opencv_test
{
namespace
{

using namespace std;
/****************************************************************************************\
 *                Test drawing works properly
\****************************************************************************************/

void runCCheckerDraw(Ptr<CChecker> pChecker, int rows, int cols, unsigned int number_of_cells_in_colorchecker)
{
    cv::Mat img(rows, cols, CV_8UC3, {0, 0, 0});

    Ptr<CCheckerDraw> cdraw = CCheckerDraw::create(pChecker);

    cdraw->draw(img);

    //make sure this contains extacly as many rectangles as in the pChecker
    vector<vector<Point>> contours;
    cv::cvtColor(img, img, COLOR_BGR2GRAY);
    findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    ASSERT_EQ(contours.size(), number_of_cells_in_colorchecker);
}

TEST(CV_mccRunCCheckerDrawTest, accuracy_MCC24)
{
    Ptr<CChecker> pChecker = CChecker::create();
    pChecker->setTarget(MCC24);
    pChecker->setBox({{0, 0}, {480, 0}, {480, 640}, {0, 640}});
    runCCheckerDraw(pChecker, 640, 480, 24);
}
TEST(CV_mccRunCCheckerDrawTest, accuracy_SG140)
{
    Ptr<CChecker> pChecker = CChecker::create();
    pChecker->setTarget(SG140);
    pChecker->setBox({{0, 0}, {480, 0}, {480, 640}, {0, 640}});
    runCCheckerDraw(pChecker, 640, 480, 140);
}
TEST(CV_mccRunCCheckerDrawTest, accuracy_VINYL18)
{
    Ptr<CChecker> pChecker = CChecker::create();
    pChecker->setTarget(VINYL18);
    pChecker->setBox({{0, 0}, {480, 0}, {480, 640}, {0, 640}});
    runCCheckerDraw(pChecker, 640, 480, 18);
}

/****************************************************************************************\
 *                Test detection works properly on the simplest images
\****************************************************************************************/

void runCCheckerDetectorBasic(std::string image_name, TYPECHART chartType)
{
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();
    std::string path = cvtest::findDataFile("mcc/" + image_name);
    cv::Mat img = imread(path);
    ASSERT_FALSE(img.empty()) << "Test image can't be loaded: " << path;

    ASSERT_TRUE(detector->process(img, chartType));
}
TEST(CV_mccRunCCheckerDetectorBasic, accuracy_SG140)
{
    runCCheckerDetectorBasic("SG140.png", SG140);
}
TEST(CV_mccRunCCheckerDetectorBasic, accuracy_MCC24)
{
    runCCheckerDetectorBasic("MCC24.png", MCC24);
}

TEST(CV_mccRunCCheckerDetectorBasic, accuracy_VINYL18)
{
    runCCheckerDetectorBasic("VINYL18.png", VINYL18);
}

TEST(CV_mcc_ccm_test, detectAndInfer1)
{
    string path = cvtest::findDataFile("mcc/10.png");
    Mat img = imread(path, IMREAD_COLOR);
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();

    // detect MCC24 board
    ASSERT_TRUE(detector->process(img, MCC24, 1, false));

    // compute CCM
    Ptr<CChecker> checker = detector->getBestColorChecker();
    Mat chartsRGB = checker->getChartsRGB();
    Mat src = chartsRGB.col(1).clone().reshape(3, chartsRGB.rows/3) / 255.;
    ColorCorrectionModel model(src, COLORCHECKER_Macbeth);
    model.run();
    Mat ccm = model.getCCM();
    //const double loss = model.getLoss();
    // compute calibrate image
    Mat calibratedImage;
    cvtColor(img, calibratedImage, COLOR_BGR2RGB);
    calibratedImage.convertTo(calibratedImage, CV_64F, 1. / 255.);
    //calibratedImage = model.infer(calibratedImage);

    Mat a;
    for (int i = 0; i < 3; i++) {
        a.release();
        a = model.infer(calibratedImage);
    }
    a.convertTo(calibratedImage, CV_8UC3, 255.);

    //calibratedImage.convertTo(calibratedImage, CV_8UC3, 255.);
    cvtColor(calibratedImage, calibratedImage, COLOR_RGB2BGR);
}

TEST(CV_mcc_ccm_test, detectAndInfer)
{
    string path = cvtest::findDataFile("mcc/mcc_ccm_test.jpg");
    Mat img = imread(path, IMREAD_COLOR);
    Ptr<CCheckerDetector> detector = CCheckerDetector::create();

    // detect MCC24 board
    ASSERT_TRUE(detector->process(img, MCC24, 1, false));

    // read gold CCM
    path = cvtest::findDataFile("mcc/mcc_ccm_test.yml");
    FileStorage fs(path, FileStorage::READ);
    ASSERT_TRUE(fs.isOpened());
    FileNode node = fs["ccm"];
    ASSERT_FALSE(node.empty());
    Mat gold_ccm;
    node >> gold_ccm;
    fs.release();

    // compute CCM
    Ptr<CChecker> checker = detector->getBestColorChecker();
    Mat chartsRGB = checker->getChartsRGB();
    Mat src = chartsRGB.col(1).clone().reshape(3, chartsRGB.rows/3) / 255.;
    ColorCorrectionModel model(src, COLORCHECKER_Macbeth);
    model.run();
    Mat ccm = model.getCCM();
    // check CCM
    ASSERT_MAT_NEAR(gold_ccm, ccm, 2e-3);

    const double gold_loss = 4.6386569120323129;
    const double loss = model.getLoss();
    // check loss
    EXPECT_NEAR(gold_loss, loss, 2e-3);

    // read gold calibrate img
    path = cvtest::findDataFile("mcc/mcc_ccm_test_res.png");
    Mat gold_img = imread(path);

    // compute calibrate image
    Mat calibratedImage;
    cvtColor(img, calibratedImage, COLOR_BGR2RGB);
    calibratedImage.convertTo(calibratedImage, CV_64F, 1. / 255.);
    calibratedImage = model.infer(calibratedImage);
    calibratedImage.convertTo(calibratedImage, CV_8UC3, 255.);
    cvtColor(calibratedImage, calibratedImage, COLOR_RGB2BGR);
    // check calibrated image
    EXPECT_MAT_NEAR(gold_img, calibratedImage, 1.);
}

} // namespace
} // namespace opencv_test
