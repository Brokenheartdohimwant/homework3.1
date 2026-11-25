#ifndef TAG_DETECTOR_HPP
#define TAG_DETECTOR_HPP

#include "HighScoreTypes.hpp"

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

// 识别黑色方形标签里的三角形 / 圆 / 方块，并给出高分区在左还是右。
//
// 流程：
//   1. 找黑色方形标签（暗区轮廓 → 四边形 → 透视矫正成 600×600）
//   2. 在矫正图里找白色三角形 → 尖角朝左还是朝右
//   3. 找红/蓝两个形状 → 圆 or 方、各自什么颜色
//   4. 按规则给出 high score:left / high score:right
class TagDetector
{
public:
    struct Params
    {
        int warpSize = 600;              // 透视矫正后的边长
        int analysisMaxSide = 900;       // 检测前把图缩到这个尺寸以内
        int darkValueMax = 90;           // V < 90 视为黑色背景
        double minTagAreaRatio = 0.005;  // 标签至少占画面的比例
        double maxSideRatio = 1.70;      // 四边形最长边 / 最短边 的上限
        int whiteSatMax = 70;            // 白色：S <= 70
        int whiteValueMin = 90;          // 白色：V >= 90
    };

    TagDetector() = default;
    explicit TagDetector(const Params& params) : params_(params) {}

    // 输入 BGR 图，输出识别结果（无论成功失败都返回结构体，用 found/decided 区分）
    TagResult detect(const cv::Mat& bgr) const;

    const Params& params() const { return params_; }

private:
    Params params_;

    // 找标签四角；失败返回 false
    bool findTagQuad(const cv::Mat& bgr, std::vector<cv::Point2f>& quad) const;

    // 透视矫正
    static cv::Mat warpTag(const cv::Mat& bgr, const std::vector<cv::Point2f>& quad, int size);

    // 白色三角形 → 尖角朝向
    bool detectTriangle(const cv::Mat& tag, TagResult& result) const;

    // 红/蓝形状
    static std::vector<ShapeBlob> detectBlobs(const cv::Mat& tag);

    // 从 blobs 里挑出圆和方块，写入颜色
    static void pickShapes(const std::vector<ShapeBlob>& blobs, TagResult& result);

    // 按规则下结论
    static void decide(TagResult& result);
};

#endif