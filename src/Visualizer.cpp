#include "Visualizer.hpp"

#include <algorithm>
#include <fstream>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

Scalar Visualizer::colorOf(ShapeColor color)
{
    switch (color)
    {
        case ShapeColor::Red:  return Scalar(0, 0, 255);
        case ShapeColor::Blue: return Scalar(255, 160, 0);
        default:               return Scalar(200, 200, 200);
    }
}

namespace
{
    // 以标签为中心向外扩一圈，裁剪出来当左半边画面
    Rect aroundTag(const Rect& rect, const Size& bounds, double ratio)
    {
        const int dx = cvRound(static_cast<double>(rect.width) * ratio);
        const int dy = cvRound(static_cast<double>(rect.height) * ratio);

        const int x0 = std::max(0, rect.x - dx);
        const int y0 = std::max(0, rect.y - dy);
        const int x1 = std::min(bounds.width, rect.x + rect.width + dx);
        const int y1 = std::min(bounds.height, rect.y + rect.height + dy);

        return Rect(x0, y0, std::max(1, x1 - x0), std::max(1, y1 - y0));
    }
}

Mat Visualizer::drawResult(const Mat& bgr, const TagResult& result) const
{
    // 左半：原始画面（裁到标签附近）+ 标签四角
    Mat left;

    if (result.tagFound && result.quad.size() == 4)
    {
        vector<Point> polygon;
        polygon.reserve(4);

        for (const auto& point : result.quad)
        {
            polygon.emplace_back(cvRound(point.x), cvRound(point.y));
        }

        const Rect roi = aroundTag(boundingRect(result.quad), bgr.size(), 0.15);

        left = bgr(roi).clone();

        for (auto& point : polygon)
        {
            point.x -= roi.x;
            point.y -= roi.y;
        }

        polylines(left, polygon, true, Scalar(0, 255, 0), 2, LINE_AA);
    }
    else
    {
        left = bgr.clone();
    }

    Mat canvas;

    // 右半：透视矫正后的标签，放大后更容易看清判定依据
    if (result.tagFound && !result.warped.empty())
    {
        Mat panel = result.warped.clone();

        // 高分区一侧淡绿
        if (result.decided)
        {
            const int half = panel.cols / 2;

            const Rect band = (result.highSide == Side::Left)
                ? Rect(0, 0, half, panel.rows)
                : Rect(half, 0, panel.cols - half, panel.rows);

            Mat tint = panel.clone();
            tint(band).setTo(Scalar(0, 190, 0));

            addWeighted(tint, 0.14, panel, 0.86, 0.0, panel);
        }

        // 白色三角形 + 尖角箭头
        if (!result.triangle.empty())
        {
            drawContours(panel, vector<vector<Point>>{result.triangle}, 0, Scalar(0, 255, 255), 2, LINE_AA);

            const Rect box = boundingRect(result.triangle);

            arrowedLine(panel,
                        Point(box.x + box.width / 2, box.y + box.height / 2),
                        Point(cvRound(result.tipPoint.x), cvRound(result.tipPoint.y)),
                        Scalar(0, 255, 255), 3, LINE_AA, 0, 0.30);
        }

        // 圆 / 方块
        for (const auto& blob : result.blobs)
        {
            const Scalar color = colorOf(blob.color);

            drawContours(panel, vector<vector<Point>>{blob.contour}, 0, color, 2, LINE_AA);

            const string label = kindName(blob.kind) + ":" + colorName(blob.color);

            putText(panel, label, Point(blob.box.x, blob.box.y + 18),
                    FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 0, 0), 3, LINE_AA);

            putText(panel, label, Point(blob.box.x, blob.box.y + 18),
                    FONT_HERSHEY_SIMPLEX, 0.5, color, 1, LINE_AA);
        }

        rectangle(panel, Rect(0, 0, panel.cols - 1, panel.rows - 1), Scalar(255, 255, 255), 2);

        // 左图缩放到与标签面板同高，再拼起来
        const double scale = static_cast<double>(panel.rows) / static_cast<double>(left.rows);

        Mat leftScaled;
        resize(left, leftScaled, Size(), scale, scale, INTER_AREA);

        hconcat(leftScaled, panel, canvas);
    }
    else
    {
        canvas = left;
    }

    // 顶部大标题：high score:left / high score:right
    const bool ok = result.decided;
    const string title = ok ? highScoreText(result.highSide) : string("searching...");
    const Scalar titleColor = ok ? Scalar(0, 255, 0) : Scalar(0, 165, 255);

    int baseline = 0;
    const Size textSize = getTextSize(title, FONT_HERSHEY_DUPLEX, 1.3, 3, &baseline);

    const Rect titleBox(8, 8, textSize.width + 26, textSize.height + baseline + 22);
    rectangle(canvas, titleBox, Scalar(0, 0, 0), FILLED);
    rectangle(canvas, titleBox, titleColor, 2);

    putText(canvas, title, Point(20, 8 + textSize.height + 10),
            FONT_HERSHEY_DUPLEX, 1.3, titleColor, 3, LINE_AA);

    // 底部说明
    const string note = ok ? result.reason : (result.tagFound ? string("tag found, ") + result.reason : result.reason);

    if (!note.empty())
    {
        const Size noteSize = getTextSize(note, FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);

        const Rect noteBox(8, canvas.rows - noteSize.height - 26,
                           noteSize.width + 20, noteSize.height + 18);

        rectangle(canvas, noteBox, Scalar(0, 0, 0), FILLED);
        putText(canvas, note, Point(noteBox.x + 10, canvas.rows - 16),
                FONT_HERSHEY_SIMPLEX, 0.55, Scalar(220, 220, 220), 1, LINE_AA);
    }

    return canvas;
}

bool Visualizer::saveImage(const Mat& image, const string& filename) const
{
    if (image.empty())
    {
        cerr << "图像为空，无法保存：" << filename << endl;
        return false;
    }

    try
    {
        if (!imwrite(filename, image))
        {
            cerr << "保存图片失败：" << filename << endl;
            return false;
        }
    }
    catch (const cv::Exception& error)
    {
        cerr << "保存图片失败：" << error.what() << endl;
        return false;
    }

    return true;
}

bool Visualizer::saveText(const TagResult& result, const string& filename) const
{
    ofstream file(filename);

    if (!file.is_open())
    {
        cerr << "无法创建结果文件：" << filename << endl;
        return false;
    }

    file << highScoreText(result.highSide) << '\n';
    file << "tag_found=" << (result.tagFound ? "true" : "false") << '\n';
    file << "decided=" << (result.decided ? "true" : "false") << '\n';
    file << "tip_side=" << sideName(result.tipSide) << '\n';
    file << "circle_color=" << colorName(result.circleColor) << '\n';
    file << "square_color=" << colorName(result.squareColor) << '\n';
    file << "same_color=" << (result.sameColor ? "true" : "false") << '\n';
    file << "tip_margin=" << result.tipMargin << '\n';
    file << "note=" << result.reason << '\n';

    return file.good();
}

