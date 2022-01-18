#include "perf_precomp.hpp"

static void initTests()
{
    cvtest::addDataSearchSubDirectory("contrib/aruco");
}

CV_PERF_TEST_MAIN(aruco, initTests())
