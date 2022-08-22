// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef __OPENCV_ARUCO_HPP__
#define __OPENCV_ARUCO_HPP__

#include "opencv2/aruco/aruco_calib_pose.hpp"
#include "opencv2/objdetect/aruco_detector.hpp"

namespace cv {
namespace aruco {

using Dictionary = ArucoDictionary;

enum CornerRefineMethod{
    CORNER_REFINE_NONE,     ///< Tag and corners detection based on the ArUco approach
    CORNER_REFINE_SUBPIX,   ///< ArUco approach and refine the corners locations using corner subpixel accuracy
    CORNER_REFINE_CONTOUR,  ///< ArUco approach and refine the corners locations using the contour-points line fitting
    CORNER_REFINE_APRILTAG, ///< Tag and corners detection based on the AprilTag 2 approach @cite wang2016iros
};

enum PREDEFINED_DICTIONARY_NAME {
    DICT_4X4_50 = 0,        ///< 4x4 bits, minimum hamming distance between any two codes = 4, 50 codes
    DICT_4X4_100,           ///< 4x4 bits, minimum hamming distance between any two codes = 3, 100 codes
    DICT_4X4_250,           ///< 4x4 bits, minimum hamming distance between any two codes = 3, 250 codes
    DICT_4X4_1000,          ///< 4x4 bits, minimum hamming distance between any two codes = 2, 1000 codes
    DICT_5X5_50,            ///< 5x5 bits, minimum hamming distance between any two codes = 8, 50 codes
    DICT_5X5_100,           ///< 5x5 bits, minimum hamming distance between any two codes = 7, 100 codes
    DICT_5X5_250,           ///< 5x5 bits, minimum hamming distance between any two codes = 6, 250 codes
    DICT_5X5_1000,          ///< 5x5 bits, minimum hamming distance between any two codes = 5, 1000 codes
    DICT_6X6_50,            ///< 6x6 bits, minimum hamming distance between any two codes = 13, 50 codes
    DICT_6X6_100,           ///< 6x6 bits, minimum hamming distance between any two codes = 12, 100 codes
    DICT_6X6_250,           ///< 6x6 bits, minimum hamming distance between any two codes = 11, 250 codes
    DICT_6X6_1000,          ///< 6x6 bits, minimum hamming distance between any two codes = 9, 1000 codes
    DICT_7X7_50,            ///< 7x7 bits, minimum hamming distance between any two codes = 19, 50 codes
    DICT_7X7_100,           ///< 7x7 bits, minimum hamming distance between any two codes = 18, 100 codes
    DICT_7X7_250,           ///< 7x7 bits, minimum hamming distance between any two codes = 17, 250 codes
    DICT_7X7_1000,          ///< 7x7 bits, minimum hamming distance between any two codes = 14, 1000 codes
    DICT_ARUCO_ORIGINAL,    ///< 6x6 bits, minimum hamming distance between any two codes = 3, 1024 codes
    DICT_APRILTAG_16h5,     ///< 4x4 bits, minimum hamming distance between any two codes = 5, 30 codes
    DICT_APRILTAG_25h9,     ///< 5x5 bits, minimum hamming distance between any two codes = 9, 35 codes
    DICT_APRILTAG_36h10,    ///< 6x6 bits, minimum hamming distance between any two codes = 10, 2320 codes
    DICT_APRILTAG_36h11     ///< 6x6 bits, minimum hamming distance between any two codes = 11, 587 codes
};


using Board = BaseArucoBoard;
using GridBoard = GridArucoBoard;
using CharucoBoard = ChArucoBoard;

CV_EXPORTS_W bool testCharucoCornersCollinear(const Ptr<CharucoBoard> &board, InputArray charucoIds);

/**
 * @brief Draw a planar board
 * @sa drawPlanarBoard
 *
 * @param board layout of the board that will be drawn. The board should be planar,
 * z coordinate is ignored
 * @param outSize size of the output image in pixels.
 * @param img output image with the board. The size of this image will be outSize
 * and the board will be on the center, keeping the board proportions.
 * @param marginSize minimum margins (in pixels) of the board in the output image
 * @param borderBits width of the marker borders.
 *
 * This function return the image of a planar board, ready to be printed. It assumes
 * the BaseArucoBoard layout specified is planar by ignoring the z coordinates of the object points.
 */
CV_EXPORTS_W void drawPlanarBoard(const Ptr<BaseArucoBoard> &board, Size outSize, OutputArray img,
                                  int marginSize = 0, int borderBits = 1);

/**
  * @brief Returns one of the predefined dictionaries defined in PREDEFINED_DICTIONARY_NAME
  */
CV_EXPORTS Ptr<Dictionary> getPredefinedDictionary(PREDEFINED_DICTIONARY_NAME name);


/**
  * @brief Returns one of the predefined dictionaries referenced by DICT_*.
  */
CV_EXPORTS_W Ptr<Dictionary> getPredefinedDictionary(int dict);


/**
  * @see generateCustomDictionary
  */
CV_EXPORTS_AS(custom_dictionary) Ptr<Dictionary> generateCustomDictionary(
        int nMarkers,
        int markerSize,
        int randomSeed=0);


/**
  * @brief Generates a new customizable marker dictionary
  *
  * @param nMarkers number of markers in the dictionary
  * @param markerSize number of bits per dimension of each markers
  * @param baseDictionary Include the markers in this dictionary at the beginning (optional)
  * @param randomSeed a user supplied seed for theRNG()
  *
  * This function creates a new dictionary composed by nMarkers markers and each markers composed
  * by markerSize x markerSize bits. If baseDictionary is provided, its markers are directly
  * included and the rest are generated based on them. If the size of baseDictionary is higher
  * than nMarkers, only the first nMarkers in baseDictionary are taken and no new marker is added.
  */
CV_EXPORTS_AS(custom_dictionary_from) Ptr<Dictionary> generateCustomDictionary(
        int nMarkers,
        int markerSize,
        const Ptr<Dictionary> &baseDictionary,
        int randomSeed=0);

/**
@deprecated Use class ArucoDetector
*/
CV_EXPORTS_W void detectMarkers(InputArray image, const Ptr<Dictionary> &dictionary, OutputArrayOfArrays corners,
                                OutputArray ids, const Ptr<DetectorParameters> &parameters = DetectorParameters::create(),
                                OutputArrayOfArrays rejectedImgPoints = noArray());

/**
@deprecated Use class ArucoDetector
*/
CV_EXPORTS_W void refineDetectedMarkers(InputArray image,const  Ptr<Board> &board,
                                        InputOutputArrayOfArrays detectedCorners,
                                        InputOutputArray detectedIds, InputOutputArrayOfArrays rejectedCorners,
                                        InputArray cameraMatrix = noArray(), InputArray distCoeffs = noArray(),
                                        float minRepDistance = 10.f, float errorCorrectionRate = 3.f,
                                        bool checkAllOrders = true, OutputArray recoveredIdxs = noArray(),
                                        const Ptr<DetectorParameters> &parameters = DetectorParameters::create());

/**
 * @brief Draw detected markers in image
 *
 * @param image input/output image. It must have 1 or 3 channels. The number of channels is not
 * altered.
 * @param corners positions of marker corners on input image.
 * (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers, the dimensions of
 * this array should be Nx4. The order of the corners should be clockwise.
 * @param ids vector of identifiers for markers in markersCorners .
 * Optional, if not provided, ids are not painted.
 * @param borderColor color of marker borders. Rest of colors (text color and first corner color)
 * are calculated based on this one to improve visualization.
 *
 * Given an array of detected marker corners and its corresponding ids, this functions draws
 * the markers in the image. The marker borders are painted and the markers identifiers if provided.
 * Useful for debugging purposes.
 *
 */
CV_EXPORTS_W void drawDetectedMarkers(InputOutputArray image, InputArrayOfArrays corners,
                                      InputArray ids = noArray(), Scalar borderColor = Scalar(0, 255, 0));

/**
 * @brief Draw a canonical marker image
 *
 * @param dictionary dictionary of markers indicating the type of markers
 * @param id identifier of the marker that will be returned. It has to be a valid id
 * in the specified dictionary.
 * @param sidePixels size of the image in pixels
 * @param img output image with the marker
 * @param borderBits width of the marker border.
 *
 * This function returns a marker image in its canonical form (i.e. ready to be printed)
 */
CV_EXPORTS_W void drawMarker(const Ptr<Dictionary> &dictionary, int id, int sidePixels, OutputArray img,
                             int borderBits = 1);

}
}

#endif
