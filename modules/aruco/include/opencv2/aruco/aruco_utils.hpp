// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#ifndef __OPENCV_ARUCO_UTILS_HPP__
#define __OPENCV_ARUCO_UTILS_HPP__

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>

namespace cv {
namespace aruco {
/**
 * @brief Copy the contents of a corners vector to an OutputArray, settings its size.
 */
static inline void _copyVector2Output(std::vector<std::vector<Point2f> > &vec, OutputArrayOfArrays out, const float scale = 1.f) {
    out.create((int)vec.size(), 1, CV_32FC2);

    if(out.isMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            Mat &m = out.getMatRef(i);
            Mat(Mat(vec[i]).t()*scale).copyTo(m);
        }
    }
    else if(out.isUMatVector()) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            UMat &m = out.getUMatRef(i);
            Mat(Mat(vec[i]).t()*scale).copyTo(m);
        }
    }
    else if(out.kind() == _OutputArray::STD_VECTOR_VECTOR){
        for (unsigned int i = 0; i < vec.size(); i++) {
            out.create(4, 1, CV_32FC2, i);
            Mat m = out.getMat(i);
            Mat(Mat(vec[i]).t()*scale).copyTo(m);
        }
    }
    else {
        CV_Error(cv::Error::StsNotImplemented,
                 "Only Mat vector, UMat vector, and vector<vector> OutputArrays are currently supported.");
    }
}
/*
static std::vector<std::vector<Point2f> > getVectors(InputArrayOfArrays& in) {
    std::vector<std::vector<Point2f> > v;
    if (in.isMatVector() || in.kind()) {
        for (size_t i = 0; i < in.total(); i++) {
            Mat m = in.getMat(i);
            std::vector<Point2f> tmp;
            m.copyTo(tmp);
            v.push_back(tmp);
        }
    }
    else if (in.isUMatVector()) {
        for (size_t i = 0; i < in.total(); i++) {
            UMat m = in.getUMat(i);
            std::vector<Point2f> tmp;
            m.copyTo(tmp);
            v.push_back(tmp);
        }
    }
    else if (in.kind() == _InputArray::STD_VECTOR_MAT) {
        for (size_t i = 0; i < in.total(); i++) {
            Mat m = in.getMat(i).reshape(2);
            std::vector<Point2f> tmp;
            m.copyTo(tmp);
            v.push_back(tmp);
        }
    }
    else {
        CV_Error(cv::Error::StsNotImplemented,
                 "Only Mat vector, UMat vector, and vector<vector> InputArrays are currently supported.");
    }
    return v;
}
*/
/**
  * @brief Convert input image to gray if it is a 3-channels image
  */
static inline void _convertToGrey(InputArray _in, OutputArray _out) {

    CV_Assert(_in.type() == CV_8UC1 || _in.type() == CV_8UC3);

    if(_in.type() == CV_8UC3)
        cvtColor(_in, _out, COLOR_BGR2GRAY);
    else
        _in.copyTo(_out);
}

}
}
#endif
