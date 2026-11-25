#ifndef HIGH_SCORE_TYPES_HPP
#define HIGH_SCORE_TYPES_HPP

#include <opencv2/opencv.hpp>

#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// 4.1 识别高分区
//
// 视觉标签 = 黑色正方形背景 + 白色三角形 + 一个圆 + 一个正方形，
// 圆和正方形各是红/蓝之一。
//
// 判定规则（和官方图例的 4+4 张样例完全一致）：
//   · 圆与方块颜色相同 → 三角形"尖角指向"的一侧是高分区
//   · 圆与方块颜色不同 → 三角形"尖角指向"的反侧是高分区
// ---------------------------------------------------------------------------

enum class ShapeColor
{
    Unknown,
    Red,
    Blue
};

enum class ShapeKind
{
    Unknown,
    Circle,
    Square
};

enum class Side
{
    Unknown,
    Left,
    Right
};

// 标签里检测到的一个彩色形状
struct ShapeBlob
{
    ShapeColor color = ShapeColor::Unknown;
    ShapeKind kind = ShapeKind::Unknown;

    cv::Rect box;
    double area = 0.0;
    double extent = 0.0;             // 轮廓面积 / 外接框面积（圆≈0.79，方≈1.0）
    cv::Point2f center{0.0f, 0.0f};
    std::vector<cv::Point> contour;
};

struct TagResult
{
    bool tagFound = false;           // 是否找到黑色方形标签
    bool decided = false;            // 是否得出 high score 结论

    std::vector<cv::Point2f> quad;   // 标签四角（TL, TR, BR, BL）

    Side tipSide = Side::Unknown;    // 三角形尖角朝向
    Side highSide = Side::Unknown;   // 最终结论
    double tipMargin = 0.0;          // 尖角判定的置信度（0~1）

    ShapeColor circleColor = ShapeColor::Unknown;
    ShapeColor squareColor = ShapeColor::Unknown;
    bool sameColor = false;

    cv::Point2f tipPoint{0.0f, 0.0f};
    std::vector<cv::Point> triangle;      // 白色三角形轮廓（warp 后的坐标）
    std::vector<ShapeBlob> blobs;

    cv::Mat warped;                  // 透视矫正后的 600×600 标签（调试用）
    std::string reason;              // 判定说明，画在图上 / 写进 txt
};

inline std::string colorName(ShapeColor color)
{
    switch (color)
    {
        case ShapeColor::Red:  return "red";
        case ShapeColor::Blue: return "blue";
        default:               return "unknown";
    }
}

inline std::string sideName(Side side)
{
    switch (side)
    {
        case Side::Left:  return "left";
        case Side::Right: return "right";
        default:          return "unknown";
    }
}

inline std::string kindName(ShapeKind kind)
{
    switch (kind)
    {
        case ShapeKind::Circle: return "circle";
        case ShapeKind::Square: return "square";
        default:                return "unknown";
    }
}

// 官方要求的输出字符串：high score:left / high score:right
inline std::string highScoreText(Side side)
{
    return "high score:" + sideName(side);
}

#endif