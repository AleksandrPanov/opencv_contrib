// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef __OPENCV_ARUCO_DETECTOR_HPP__
#define __OPENCV_ARUCO_DETECTOR_HPP__
#include <opencv2/aruco/board.hpp>
#include <opencv2/aruco/dictionary.hpp>

namespace cv {
namespace aruco {

//! @addtogroup aruco
//! @{

enum CornerRefineMethod{
    CORNER_REFINE_NONE,     ///< Tag and corners detection based on the ArUco approach
    CORNER_REFINE_SUBPIX,   ///< ArUco approach and refine the corners locations using corner subpixel accuracy
    CORNER_REFINE_CONTOUR,  ///< ArUco approach and refine the corners locations using the contour-points line fitting
    CORNER_REFINE_APRILTAG, ///< Tag and corners detection based on the AprilTag 2 approach @cite wang2016iros
};

/**
 * @brief Parameters for the detectMarker process:
 * - adaptiveThreshWinSizeMin: minimum window size for adaptive thresholding before finding
 *   contours (default 3).
 * - adaptiveThreshWinSizeMax: maximum window size for adaptive thresholding before finding
 *   contours (default 23).
 * - adaptiveThreshWinSizeStep: increments from adaptiveThreshWinSizeMin to adaptiveThreshWinSizeMax
 *   during the thresholding (default 10).
 * - adaptiveThreshConstant: constant for adaptive thresholding before finding contours (default 7)
 * - minMarkerPerimeterRate: determine minimum perimeter for marker contour to be detected. This
 *   is defined as a rate respect to the maximum dimension of the input image (default 0.03).
 * - maxMarkerPerimeterRate:  determine maximum perimeter for marker contour to be detected. This
 *   is defined as a rate respect to the maximum dimension of the input image (default 4.0).
 * - polygonalApproxAccuracyRate: minimum accuracy during the polygonal approximation process to
 *   determine which contours are squares. (default 0.03)
 * - minCornerDistanceRate: minimum distance between corners for detected markers relative to its
 *   perimeter (default 0.05)
 * - minDistanceToBorder: minimum distance of any corner to the image border for detected markers
 *   (in pixels) (default 3)
 * - minMarkerDistanceRate: minimum mean distance beetween two marker corners to be considered
 *   similar, so that the smaller one is removed. The rate is relative to the smaller perimeter
 *   of the two markers (default 0.05).
 * - cornerRefinementMethod: corner refinement method. (CORNER_REFINE_NONE, no refinement.
 *   CORNER_REFINE_SUBPIX, do subpixel refinement. CORNER_REFINE_CONTOUR use contour-Points,
 *   CORNER_REFINE_APRILTAG  use the AprilTag2 approach). (default CORNER_REFINE_NONE)
 * - cornerRefinementWinSize: window size for the corner refinement process (in pixels) (default 5).
 * - cornerRefinementMaxIterations: maximum number of iterations for stop criteria of the corner
 *   refinement process (default 30).
 * - cornerRefinementMinAccuracy: minimum error for the stop cristeria of the corner refinement
 *   process (default: 0.1)
 * - markerBorderBits: number of bits of the marker border, i.e. marker border width (default 1).
 * - perspectiveRemovePixelPerCell: number of bits (per dimension) for each cell of the marker
 *   when removing the perspective (default 4).
 * - perspectiveRemoveIgnoredMarginPerCell: width of the margin of pixels on each cell not
 *   considered for the determination of the cell bit. Represents the rate respect to the total
 *   size of the cell, i.e. perspectiveRemovePixelPerCell (default 0.13)
 * - maxErroneousBitsInBorderRate: maximum number of accepted erroneous bits in the border (i.e.
 *   number of allowed white bits in the border). Represented as a rate respect to the total
 *   number of bits per marker (default 0.35).
 * - minOtsuStdDev: minimun standard deviation in pixels values during the decodification step to
 *   apply Otsu thresholding (otherwise, all the bits are set to 0 or 1 depending on mean higher
 *   than 128 or not) (default 5.0)
 * - errorCorrectionRate error correction rate respect to the maximun error correction capability
 *   for each dictionary. (default 0.6).
 * - aprilTagMinClusterPixels: reject quads containing too few pixels. (default 5)
 * - aprilTagMaxNmaxima: how many corner candidates to consider when segmenting a group of pixels into a quad. (default 10)
 * - aprilTagCriticalRad: Reject quads where pairs of edges have angles that are close to straight or close to
 *   180 degrees. Zero means that no quads are rejected. (In radians) (default 10*PI/180)
 * - aprilTagMaxLineFitMse:  When fitting lines to the contours, what is the maximum mean squared error
 *   allowed?  This is useful in rejecting contours that are far from being quad shaped; rejecting
 *   these quads "early" saves expensive decoding processing. (default 10.0)
 * - aprilTagMinWhiteBlackDiff: When we build our model of black & white pixels, we add an extra check that
 *   the white model must be (overall) brighter than the black model.  How much brighter? (in pixel values, [0,255]). (default 5)
 * - aprilTagDeglitch:  should the thresholded image be deglitched? Only useful for very noisy images. (default 0)
 * - aprilTagQuadDecimate: Detection of quads can be done on a lower-resolution image, improving speed at a
 *   cost of pose accuracy and a slight decrease in detection rate. Decoding the binary payload is still
 *   done at full resolution. (default 0.0)
 * - aprilTagQuadSigma: What Gaussian blur should be applied to the segmented image (used for quad detection?)
 *   Parameter is the standard deviation in pixels.  Very noisy images benefit from non-zero values (e.g. 0.8). (default 0.0)
 * - detectInvertedMarker: to check if there is a white marker. In order to generate a "white" marker just
 *   invert a normal marker by using a tilde, ~markerImage. (default false)
 * - useAruco3Detection: to enable the new and faster Aruco detection strategy. The most important observation from the authors of
 *   Romero-Ramirez et al: Speeded up detection of squared fiducial markers (2018) is, that the binary
 *   code of a marker can be reliably detected if the canonical image (that is used to extract the binary code)
 *   has a size of minSideLengthCanonicalImg (in practice tau_c=16-32 pixels).
 *   Link to article: https://www.researchgate.net/publication/325787310_Speeded_Up_Detection_of_Squared_Fiducial_Markers
 *   In addition, very small markers are barely useful for pose estimation and thus a we can define a minimum marker size that we
 *   still want to be able to detect (e.g. 50x50 pixel).
 *   To decouple this from the initial image size they propose to resize the input image
 *   to (I_w_r, I_h_r) = (tau_c / tau_dot_i) * (I_w, I_h), with tau_dot_i = tau_c + max(I_w,I_h) * tau_i.
 *   Here tau_i (parameter: minMarkerLengthRatioOriginalImg) is a ratio in the range [0,1].
 *   If we set this to 0, the smallest marker we can detect
 *   has a side length of tau_c. If we set it to 1 the marker would fill the entire image.
 *   For a FullHD video a good value to start with is 0.1.
 * - minSideLengthCanonicalImg: minimum side length of a marker in the canonical image.
 *   Latter is the binarized image in which contours are searched.
 *   So all contours with a size smaller than minSideLengthCanonicalImg*minSideLengthCanonicalImg will omitted from the search.
 * - minMarkerLengthRatioOriginalImg:  range [0,1], eq (2) from paper
 *   The parameter tau_i has a direct influence on the processing speed.
 */
struct CV_EXPORTS_W DetectorParameters {
    DetectorParameters():
        adaptiveThreshWinSizeMin(3),
        adaptiveThreshWinSizeMax(23),
        adaptiveThreshWinSizeStep(10),
        adaptiveThreshConstant(7),
        minMarkerPerimeterRate(0.03),
        maxMarkerPerimeterRate(4.),
        polygonalApproxAccuracyRate(0.03),
        minCornerDistanceRate(0.05),
        minDistanceToBorder(3),
        minMarkerDistanceRate(0.05),
        cornerRefinementMethod(CORNER_REFINE_NONE),
        cornerRefinementWinSize(5),
        cornerRefinementMaxIterations(30),
        cornerRefinementMinAccuracy(0.1),
        markerBorderBits(1),
        perspectiveRemovePixelPerCell(4),
        perspectiveRemoveIgnoredMarginPerCell(0.13),
        maxErroneousBitsInBorderRate(0.35),
        minOtsuStdDev(5.0),
        errorCorrectionRate(0.6),
        aprilTagQuadDecimate(0.0),
        aprilTagQuadSigma(0.0),
        aprilTagMinClusterPixels(5),
        aprilTagMaxNmaxima(10),
        aprilTagCriticalRad( (float)(10* CV_PI /180) ),
        aprilTagMaxLineFitMse(10.0),
        aprilTagMinWhiteBlackDiff(5),
        aprilTagDeglitch(0),
        detectInvertedMarker(false),
        useAruco3Detection(false),
        minSideLengthCanonicalImg(32),
        minMarkerLengthRatioOriginalImg(0.0) {};

/**
  * @brief Create a new set of DetectorParameters with default values.
  */
    CV_WRAP static Ptr<DetectorParameters> create() {
        Ptr<DetectorParameters> params = makePtr<DetectorParameters>();
        return params;
}

    CV_WRAP bool readDetectorParameters(const FileNode& fn);

    CV_PROP_RW int adaptiveThreshWinSizeMin;
    CV_PROP_RW int adaptiveThreshWinSizeMax;
    CV_PROP_RW int adaptiveThreshWinSizeStep;
    CV_PROP_RW double adaptiveThreshConstant;
    CV_PROP_RW double minMarkerPerimeterRate;
    CV_PROP_RW double maxMarkerPerimeterRate;
    CV_PROP_RW double polygonalApproxAccuracyRate;
    CV_PROP_RW double minCornerDistanceRate;
    CV_PROP_RW int minDistanceToBorder;
    CV_PROP_RW double minMarkerDistanceRate;
    CV_PROP_RW int cornerRefinementMethod;
    CV_PROP_RW int cornerRefinementWinSize;
    CV_PROP_RW int cornerRefinementMaxIterations;
    CV_PROP_RW double cornerRefinementMinAccuracy;
    CV_PROP_RW int markerBorderBits;
    CV_PROP_RW int perspectiveRemovePixelPerCell;
    CV_PROP_RW double perspectiveRemoveIgnoredMarginPerCell;
    CV_PROP_RW double maxErroneousBitsInBorderRate;
    CV_PROP_RW double minOtsuStdDev;
    CV_PROP_RW double errorCorrectionRate;

    // April :: User-configurable parameters.
    CV_PROP_RW float aprilTagQuadDecimate;
    CV_PROP_RW float aprilTagQuadSigma;

    // April :: Internal variables
    CV_PROP_RW int aprilTagMinClusterPixels;
    CV_PROP_RW int aprilTagMaxNmaxima;
    CV_PROP_RW float aprilTagCriticalRad;
    CV_PROP_RW float aprilTagMaxLineFitMse;
    CV_PROP_RW int aprilTagMinWhiteBlackDiff;
    CV_PROP_RW int aprilTagDeglitch;

    // to detect white (inverted) markers
    CV_PROP_RW bool detectInvertedMarker;

    // New Aruco functionality proposed in the paper:
    // Romero-Ramirez et al: Speeded up detection of squared fiducial markers (2018)
    CV_PROP_RW bool useAruco3Detection;
    CV_PROP_RW int minSideLengthCanonicalImg;
    CV_PROP_RW float minMarkerLengthRatioOriginalImg;
};

struct CV_EXPORTS_W RefineParameters {
    RefineParameters() {
        minRepDistance = 10.f;
        errorCorrectionRate = 3.f;
        checkAllOrders = true;
    }

    RefineParameters(float _minRepDistance, float _errorCorrectionRate, bool _checkAllOrders):
    minRepDistance(_minRepDistance), errorCorrectionRate(_errorCorrectionRate), checkAllOrders(_checkAllOrders) {}

    CV_WRAP static Ptr<RefineParameters> create() {
        return makePtr<RefineParameters>();
    }

    CV_WRAP static Ptr<RefineParameters> create(float _minRepDistance, float _errorCorrectionRate, bool _checkAllOrders) {
        return makePtr<RefineParameters>(_minRepDistance, _errorCorrectionRate, _checkAllOrders);
    }

    /// minRepDistance minimum distance between the corners of the rejected candidate and the reprojected marker in
    /// order to consider it as a correspondence.
    CV_PROP_RW float minRepDistance;
    /// minRepDistance rate of allowed erroneous bits respect to the error correction
    /// capability of the used dictionary. -1 ignores the error correction step.
    CV_PROP_RW float errorCorrectionRate;
    /// checkAllOrders consider the four posible corner orders in the rejectedCorners array.
    // * If it set to false, only the provided corner order is considered (default true).
    CV_PROP_RW bool checkAllOrders;
};

class CV_EXPORTS_W ArucoDetector
{
public:
    /// dictionary indicates the type of markers that will be searched
    CV_PROP_RW Ptr<Dictionary> dictionary;
    /// parameters marker detection parameters
    CV_PROP_RW Ptr<DetectorParameters> params;
    /// refineParams marker refine parameters
    CV_PROP_RW Ptr<RefineParameters> refineParams;

    ArucoDetector(const Ptr<Dictionary> &_dictionary = getPredefinedDictionary(DICT_4X4_50), const Ptr<DetectorParameters> &_params = DetectorParameters::create(),
                  const Ptr<RefineParameters> &_refineParams = RefineParameters::create()):
                  dictionary(_dictionary), params(_params), refineParams(_refineParams) {}

    CV_WRAP static Ptr<ArucoDetector> create(const Ptr<Dictionary> &_dictionary, const Ptr<DetectorParameters> &_params) {
        return makePtr<ArucoDetector>(_dictionary, _params);
    }

    /**
     * @brief Basic marker detection
     *
     * @param image input image
     * @param corners vector of detected marker corners. For each marker, its four corners
     * are provided, (e.g std::vector<std::vector<cv::Point2f> > ). For N detected markers,
     * the dimensions of this array is Nx4. The order of the corners is clockwise.
     * @param ids vector of identifiers of the detected markers. The identifier is of type int
     * (e.g. std::vector<int>). For N detected markers, the size of ids is also N.
     * The identifiers have the same order than the markers in the imgPoints array.
     * @param rejectedImgPoints contains the imgPoints of those squares whose inner code has not a
     * correct codification. Useful for debugging purposes.
     *
     * Performs marker detection in the input image. Only markers included in the specific dictionary
     * are searched. For each detected marker, it returns the 2D position of its corner in the image
     * and its corresponding identifier.
     * Note that this function does not perform pose estimation.
     * @note The function does not correct lens distortion or takes it into account. It's recommended to undistort
     * input image with corresponging camera model, if camera parameters are known
     * @sa undistort, estimatePoseSingleMarkers,  estimatePoseBoard
     *
     */
    CV_EXPORTS_W void detectMarkers(InputArray image, OutputArrayOfArrays corners, OutputArray ids,
                                    OutputArrayOfArrays rejectedImgPoints = noArray());

    /**
     * @brief Refind not detected markers based on the already detected and the board layout
     *
     * @param image input image
     * @param board layout of markers in the board.
     * @param detectedCorners vector of already detected marker corners.
     * @param detectedIds vector of already detected marker identifiers.
     * @param rejectedCorners vector of rejected candidates during the marker detection process.
     * @param cameraMatrix optional input 3x3 floating-point camera matrix
     * \f$A = \vecthreethree{f_x}{0}{c_x}{0}{f_y}{c_y}{0}{0}{1}\f$
     * @param distCoeffs optional vector of distortion coefficients
     * \f$(k_1, k_2, p_1, p_2[, k_3[, k_4, k_5, k_6],[s_1, s_2, s_3, s_4]])\f$ of 4, 5, 8 or 12 elements
     * @param recoveredIdxs Optional array to returns the indexes of the recovered candidates in the
     * original rejectedCorners array.
     *
     * This function tries to find markers that were not detected in the basic detecMarkers function.
     * First, based on the current detected marker and the board layout, the function interpolates
     * the position of the missing markers. Then it tries to find correspondence between the reprojected
     * markers and the rejected candidates based on the minRepDistance and errorCorrectionRate
     * parameters.
     * If camera parameters and distortion coefficients are provided, missing markers are reprojected
     * using projectPoint function. If not, missing marker projections are interpolated using global
     * homography, and all the marker corners in the board must have the same Z coordinate.
     */
    CV_EXPORTS_W void refineDetectedMarkers(InputArray image, const Ptr<Board> &board,
                                            InputOutputArrayOfArrays detectedCorners,
                                            InputOutputArray detectedIds, InputOutputArrayOfArrays rejectedCorners,
                                            InputArray cameraMatrix = noArray(), InputArray distCoeffs = noArray(),
                                            OutputArray recoveredIdxs = noArray());
};

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

//! @}

}
}

#endif
