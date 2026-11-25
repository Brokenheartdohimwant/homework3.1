#include "TagDetector.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <utility>
#include <vector>

namespace
{
    // 四个角排成 左上、右上、右下、左下
    std::vector<cv::Point2f> orderQuad(const std::vector<cv::Point2f>& points)
    {
        std::vector<cv::Point2f> ordered;
        ordered.reserve(4);

        std::vector<size_t> index = {0, 1, 2, 3};

        std::sort(index.begin(), index.end(), [&](size_t a, size_t b)
        {
            return (points[a].x + points[a].y) < (points[b].x + points[b].y);
        });

        const cv::Point2f topLeft = points[index[0]];
        const cv::Point2f bottomRight = points[index[3]];

        // 剩下两个用 y-x 区分：右上更小，左下更大
        const cv::Point2f a = points[index[1]];
        const cv::Point2f b = points[index[2]];

        const cv::Point2f topRight = (a.y - a.x) < (b.y - b.x) ? a : b;
        const cv::Point2f bottomLeft = (a.y - a.x) < (b.y - b.x) ? b : a;

        ordered.push_back(topLeft);
        ordered.push_back(topRight);
        ordered.push_back(bottomRight);
        ordered.push_back(bottomLeft);

        return ordered;
    }

    // 把图四周的一圈置 0，避免白纸边缘混进掩膜
    void clearBorder(cv::Mat& mask, double ratio)
    {
        const int band = std::max(1, cvRound(static_cast<double>(mask.cols) * ratio));

        if (2 * band >= mask.rows || 2 * band >= mask.cols)
        {
            return;
        }

        mask.rowRange(0, band).setTo(cv::Scalar(0));
        mask.rowRange(mask.rows - band, mask.rows).setTo(cv::Scalar(0));
        mask.colRange(0, band).setTo(cv::Scalar(0));
        mask.colRange(mask.cols - band, mask.cols).setTo(cv::Scalar(0));
    }
}

TagResult TagDetector::detect(const cv::Mat& bgr) const
{
    TagResult result;

    if (bgr.empty())
    {
        result.reason = "输入图像为空";
        return result;
    }

    // 1. 缩到分析尺寸，加快速度
    cv::Mat small = bgr;
    const int longest = std::max(bgr.cols, bgr.rows);

    if (longest > params_.analysisMaxSide)
    {
        const double scale = static_cast<double>(params_.analysisMaxSide) / static_cast<double>(longest);
        cv::resize(bgr, small, cv::Size(), scale, scale, cv::INTER_AREA);
    }

    // 2. 找黑色方形标签
    std::vector<cv::Point2f> quad;

    if (!findTagQuad(small, quad))
    {
        result.reason = "没有找到黑色方形标签";
        return result;
    }

    result.tagFound = true;

    // 3. 透视矫正（用分析尺寸下的角点）
    const cv::Mat warped = warpTag(small, quad, params_.warpSize);
    result.warped = warped;

    // 角点换算回原图坐标，方便在原图上标注
    const double backScale = static_cast<double>(bgr.cols) / static_cast<double>(small.cols);

    result.quad = quad;

    for (auto& point : result.quad)
    {
        point.x = static_cast<float>(point.x * backScale);
        point.y = static_cast<float>(point.y * backScale);
    }

    // 4. 白三角 / 彩色形状 / 规则判定
    detectTriangle(warped, result);
    result.blobs = detectBlobs(warped);
    pickShapes(result.blobs, result);
    decide(result);

    return result;
}

bool TagDetector::findTagQuad(const cv::Mat& bgr, std::vector<cv::Point2f>& quad) const
{
    cv::Mat hsv;
    cv::cvtColor(bgr, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);   // H, S, V

    cv::Mat dark;
    cv::inRange(channels[2], cv::Scalar(0), cv::Scalar(params_.darkValueMax), dark);

    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
    cv::morphologyEx(dark, dark, cv::MORPH_OPEN, kernel);
    cv::morphologyEx(dark, dark, cv::MORPH_CLOSE, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dark, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    const double minArea = static_cast<double>(bgr.total()) * params_.minTagAreaRatio;

    double bestArea = 0.0;
    std::vector<cv::Point2f> best;

    for (const auto& contour : contours)
    {
        const double area = cv::contourArea(contour);

        if (area < minArea)
        {
            continue;
        }

        // 先试四边形拟合；失败就用最小外接旋转矩形
        const double perimeter = cv::arcLength(contour, true);

        std::vector<cv::Point> approx;
        cv::approxPolyDP(contour, approx, 0.03 * perimeter, true);

        std::vector<cv::Point2f> candidate;

        if (approx.size() == 4 && cv::isContourConvex(approx))
        {
            for (const auto& point : approx)
            {
                candidate.emplace_back(static_cast<float>(point.x), static_cast<float>(point.y));
            }
        }
        else
        {
            const cv::RotatedRect box = cv::minAreaRect(contour);

            cv::Point2f corners[4];
            box.points(corners);

            candidate.assign(corners, corners + 4);
        }

        // 方形程度：最长边 / 最短边
        double minSide = 1e9;
        double maxSide = 0.0;

        for (int i = 0; i < 4; ++i)
        {
            const cv::Point2f& a = candidate[static_cast<size_t>(i)];
            const cv::Point2f& b = candidate[static_cast<size_t>((i + 1) % 4)];

            const double dx = static_cast<double>(a.x - b.x);
            const double dy = static_cast<double>(a.y - b.y);
            const double length = std::hypot(dx, dy);

            minSide = std::min(minSide, length);
            maxSide = std::max(maxSide, length);
        }

        if (minSide < 1.0 || maxSide / minSide > params_.maxSideRatio)
        {
            continue;
        }

        // 填充率：轮廓面积要接近四边形面积（排除空心框）
        const double quadArea = std::fabs(cv::contourArea(candidate));

        if (quadArea <= 0.0 || area / quadArea < 0.75)
        {
            continue;
        }

        if (area > bestArea)
        {
            bestArea = area;
            best = orderQuad(candidate);
        }
    }

    if (best.empty())
    {
        return false;
    }

    quad = best;

    return true;
}

cv::Mat TagDetector::warpTag(const cv::Mat& bgr, const std::vector<cv::Point2f>& quad, int size)
{
    const float last = static_cast<float>(size - 1);

    const cv::Point2f target[4] = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(last, 0.0f),
        cv::Point2f(last, last),
        cv::Point2f(0.0f, last)
    };

    const cv::Mat transform = cv::getPerspectiveTransform(quad.data(), target);

    cv::Mat warped;
    cv::warpPerspective(bgr, warped, transform, cv::Size(size, size));

    return warped;
}

bool TagDetector::detectTriangle(const cv::Mat& tag, TagResult& result) const
{
    cv::Mat hsv;
    cv::cvtColor(tag, hsv, cv::COLOR_BGR2HSV);

    std::vector<cv::Mat> channels;
    cv::split(hsv, channels);   // H, S, V

    cv::Mat lowSaturation;
    cv::Mat bright;

    cv::inRange(channels[1], cv::Scalar(0), cv::Scalar(params_.whiteSatMax), lowSaturation);
    cv::inRange(channels[2], cv::Scalar(params_.whiteValueMin), cv::Scalar(255), bright);

    cv::Mat white;
    cv::bitwise_and(lowSaturation, bright, white);

    // 注意：必须清掉边框再开运算，否则贴边的白色噪点会把外接框撑到角上
    clearBorder(white, 0.04);

    // 3×3 开运算两遍，去掉屏幕抖动/JPEG 产生的小白点
    // （输出写进新 Mat，避免 in-place 改坏后面的统计）
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    cv::Mat opened;
    cv::morphologyEx(white, opened, cv::MORPH_OPEN, kernel);

    cv::Mat cleaned;
    cv::morphologyEx(opened, cleaned, cv::MORPH_OPEN, kernel);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(cleaned, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty())
    {
        result.reason = "没有找到白色三角形";
        return false;
    }

    // 取面积最大的白色区域（三角形）
    size_t bestIndex = 0;
    double bestArea = 0.0;

    for (size_t i = 0; i < contours.size(); ++i)
    {
        const double area = cv::contourArea(contours[i]);

        if (area > bestArea)
        {
            bestArea = area;
            bestIndex = i;
        }
    }

    if (bestArea < static_cast<double>(tag.total()) * 0.002)
    {
        result.reason = "白色三角形太小";
        return false;
    }

    result.triangle = contours[bestIndex];

    // -----------------------------------------------------------------------
    // 只保留三角形本体
    //
    // 关键：白色掩膜里除了三角形还有零星白点（屏幕抖动、JPEG 噪点）。如果直接
    // 对整个掩膜求外接框，会把框一路撑到角落，再按中线一切，左右面积就被稀释
    // 成几乎相等（实测 17750 vs 17984，只剩 0.6% 差异），方向随机错。
    // 所以先把三角形按填充画进一张单独的 mask，后面只统计这一块。
    // -----------------------------------------------------------------------
    cv::Mat triangleMask = cv::Mat::zeros(white.size(), white.type());
    cv::drawContours(triangleMask, contours, static_cast<int>(bestIndex), cv::Scalar(255), cv::FILLED);

    // -----------------------------------------------------------------------
    // 尖角朝向：比较三角形"最左列"和"最右列"的白色高度
    //
    // 三角形 = 一侧几乎贯穿全高（底边），另一侧只有一两个像素（尖角）。
    // 所以：两边各取一条窄带，白色像素多的那侧是底边，尖角朝另一侧。
    //
    // 为什么不用别的办法（都用官方 8 种标签 × 6 种变形实测过）：
    //   · 左右两半像素数：只差 0.4%，0.37% 的扰动就翻转 —— 现场就是这么错判的
    //   · 重心 vs 外接框中心：32/48
    //   · "两边面积/占位"：34/48（被彩色形状和遮挡干扰）
    //   · 最白列（底边）：0/48
    //   · 两侧高度：48/48 ✅
    // -----------------------------------------------------------------------
    const cv::Rect box = cv::boundingRect(result.triangle);
    const int stripWidth = std::max(3, box.width / 20);

    const int leftHeight = cv::countNonZero(
        triangleMask.colRange(box.x, std::min(box.x + stripWidth, box.x + box.width))
            .rowRange(box.y, box.y + box.height));

    const int rightHeight = cv::countNonZero(
        triangleMask.colRange(std::max(box.x + box.width - stripWidth, box.x), box.x + box.width)
            .rowRange(box.y, box.y + box.height));

    if (leftHeight + rightHeight <= 0)
    {
        result.reason = "白色三角形像素统计失败";
        return false;
    }

    // 高的一侧是底边，尖角朝另一边
    result.tipSide = (leftHeight > rightHeight) ? Side::Right : Side::Left;

    const double imbalance = std::abs(leftHeight - rightHeight) /
                             static_cast<double>(leftHeight + rightHeight);

    result.tipMargin = std::min(1.0, imbalance);   // 越大越有把握（正常 >0.9）

    // 尖角点：朝向那一侧、三角形最靠外的一列的中点
    const int midX = box.x + box.width / 2;

    int tipColumn = (result.tipSide == Side::Left) ? box.x : box.x + box.width - 1;

    const int step = (result.tipSide == Side::Left) ? 1 : -1;

    for (int c = tipColumn; (step > 0 ? c < midX : c >= midX); c += step)
    {
        if (cv::countNonZero(triangleMask.col(c)) > 0)
        {
            tipColumn = c;
            break;
        }
    }

    int sumY = 0;
    int countY = 0;

    for (int r = box.y; r < box.y + box.height; ++r)
    {
        if (triangleMask.at<unsigned char>(r, tipColumn) != 0)
        {
            sumY += r;
            ++countY;
        }
    }

    result.tipPoint = cv::Point2f(
        static_cast<float>(tipColumn),
        countY > 0 ? static_cast<float>(sumY) / static_cast<float>(countY)
                   : static_cast<float>(box.y + box.height / 2));

    return true;
}

std::vector<ShapeBlob> TagDetector::detectBlobs(const cv::Mat& tag)
{
    std::vector<ShapeBlob> blobs;

    cv::Mat hsv;
    cv::cvtColor(tag, hsv, cv::COLOR_BGR2HSV);

    // 红色跨 0 度，分两段
    cv::Mat redA;
    cv::Mat redB;
    cv::Mat red;

    cv::inRange(hsv, cv::Scalar(0, 110, 45), cv::Scalar(12, 255, 255), redA);
    cv::inRange(hsv, cv::Scalar(168, 110, 45), cv::Scalar(179, 255, 255), redB);
    cv::bitwise_or(redA, redB, red);

    cv::Mat blue;
    cv::inRange(hsv, cv::Scalar(85, 110, 45), cv::Scalar(115, 255, 255), blue);

    const std::pair<ShapeColor, cv::Mat> masks[2] = {
        {ShapeColor::Red, red},
        {ShapeColor::Blue, blue}
    };

    const double minArea = static_cast<double>(tag.total()) * 0.004;

    for (const auto& item : masks)
    {
        cv::Mat mask = item.second.clone();
        clearBorder(mask, 0.04);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (const auto& contour : contours)
        {
            const double area = cv::contourArea(contour);

            if (area < minArea)
            {
                continue;
            }

            const cv::Rect box = cv::boundingRect(contour);

            if (box.width < 8 || box.height < 8)
            {
                continue;
            }

            const double extent = area / static_cast<double>(box.width * box.height);

            std::vector<cv::Point> approx;
            cv::approxPolyDP(contour, approx, 0.03 * cv::arcLength(contour, true), true);

            ShapeBlob blob;
            blob.color = item.first;
            blob.area = area;
            blob.box = box;
            blob.extent = extent;
            blob.center = cv::Point2f(
                static_cast<float>(box.x) + static_cast<float>(box.width) * 0.5f,
                static_cast<float>(box.y) + static_cast<float>(box.height) * 0.5f);
            blob.contour = contour;

            // 圆：外接框填充率 ≈ 0.79，拟合点多；方：填充率 ≈ 1.0，拟合点 4 个
            blob.kind = (approx.size() >= 5 || extent < 0.87) ? ShapeKind::Circle : ShapeKind::Square;

            blobs.push_back(blob);
        }
    }

    return blobs;
}

void TagDetector::pickShapes(const std::vector<ShapeBlob>& blobs, TagResult& result)
{
    const ShapeBlob* circle = nullptr;
    const ShapeBlob* square = nullptr;

    for (const auto& blob : blobs)
    {
        if (blob.kind == ShapeKind::Circle &&
            (circle == nullptr || blob.area > circle->area))
        {
            circle = &blob;
        }

        if (blob.kind == ShapeKind::Square &&
            (square == nullptr || blob.area > square->area))
        {
            square = &blob;
        }
    }

    // 兜底：官方布局固定为"圆在上、方在下"
    if ((circle == nullptr || square == nullptr) && blobs.size() >= 2)
    {
        std::vector<const ShapeBlob*> byHeight;

        for (const auto& blob : blobs)
        {
            byHeight.push_back(&blob);
        }

        std::sort(byHeight.begin(), byHeight.end(), [](const ShapeBlob* a, const ShapeBlob* b)
        {
            return a->center.y < b->center.y;
        });

        circle = byHeight.front();
        square = byHeight.back();
    }

    if (circle != nullptr)
    {
        result.circleColor = circle->color;
    }

    if (square != nullptr)
    {
        result.squareColor = square->color;
    }
}

void TagDetector::decide(TagResult& result)
{
    if (result.tipSide == Side::Unknown)
    {
        result.reason = "三角形朝向未识别";
        return;
    }

    if (result.circleColor == ShapeColor::Unknown || result.squareColor == ShapeColor::Unknown)
    {
        result.reason = "圆形或方块颜色未识别";
        return;
    }

    result.sameColor = (result.circleColor == result.squareColor);

    if (result.sameColor)
    {
        // 颜色相同：尖角指向的一侧
        result.highSide = result.tipSide;
    }
    else
    {
        // 颜色不同：尖角指向的反侧
        result.highSide = (result.tipSide == Side::Left) ? Side::Right : Side::Left;
    }

    result.decided = true;

    result.reason =
        std::string("circle=") + colorName(result.circleColor) +
        "  square=" + colorName(result.squareColor) +
        (result.sameColor ? "  (same)" : "  (diff)") +
        "  tip=" + sideName(result.tipSide);
}





