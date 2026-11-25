#ifndef VISUALIZER_HPP
#define VISUALIZER_HPP

#include "HighScoreTypes.hpp"

#include <opencv2/opencv.hpp>

#include <string>

// 把识别结果画到图上，并保存 png / txt
class Visualizer
{
public:
    // 画面顶部大字显示 high score:left / high score:right
    cv::Mat drawResult(const cv::Mat& bgr, const TagResult& result) const;

    bool saveImage(const cv::Mat& image, const std::string& filename) const;

    // 第一行就是官方要求的 high score:xxx
    bool saveText(const TagResult& result, const std::string& filename) const;

private:
    static cv::Scalar colorOf(ShapeColor color);
};

#endif
