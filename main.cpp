#include "HighScoreTypes.hpp"
#include "ImageSource.hpp"
#include "TagDetector.hpp"
#include "SerialLink.hpp"
#include "Visualizer.hpp"

#include <opencv2/opencv.hpp>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

using namespace std;

// ---------------------------------------------------------------------------
// 命令行
// ---------------------------------------------------------------------------

struct Options
{
    enum class Source
    {
        Camera,
        Image
    };

    Source source = Source::Camera;

    string value;                 // 图片路径
    string outDir = "output";
    int cameraIndex = 0;

    int votes = 4;                // 最近 5 帧里命中几帧才锁定
    bool autoSave = false;        // 锁定后自动保存，不用按键
    bool fast = false;            // 小图快跑（640），低配虚拟机用

    string serialPort;            // 例如 /dev/ttyUSB0；空 = 不发串口
    int serialBaud = 115200;
    bool fireAfterSend = false;   // 发完 L/R 后自动再发 F（开火）

    bool showWindow = true;
    bool debug = false;           // 额外保存透视矫正后的标签图
    bool help = false;
};

void printUsage(const char* program)
{
    const string p = program;

    cout
        << "用法:\n"
        << "  " << p << "                       用摄像头实时识别（默认方式）\n"
        << "  " << p << " --image tag.png        识别一张图片\n"
        << "\n"
        << "可选参数:\n"
        << "  --camera N      摄像头编号，默认 0\n"
        << "  --out DIR       输出目录，默认 output\n"
        << "  --votes N       连续一致多少帧才锁定，默认 4（最近 5 帧）\n"
        << "  --auto          锁定后自动保存，不用按空格\n"
        << "  --fast          用 640 的分析尺寸跑，低配虚拟机更流畅\n"
        << "  --serial PORT   识别出结果后通过串口发给单片机，例如 /dev/ttyUSB0\n"
        << "  --baud N        串口波特率，默认 115200\n"
        << "  --fire          发完 L/R 后自动再发 F（开火）\n"
        << "  --no-window     不弹出窗口（没有图形界面时使用）\n"
        << "  --debug         额外保存透视矫正后的标签图（output/tag_warped.png）\n"
        << "  -h, --help      显示本帮助\n"
        << "\n"
        << "摄像头模式按键:\n"
        << "  [空格]  锁定后保存并退出\n"
        << "  [ESC]   锁定时解除锁定回到实时画面；未锁定时退出\n"
        << "  [q]     直接退出\n"
        << "\n"
        << "输出:\n"
        << "  output/result.png   识别结果截图\n"
        << "  output/result.txt   第一行就是 high score:left / high score:right\n"
        << "\n"
        << "判定规则（与官方图例一致）:\n"
        << "  圆与方块颜色相同  -> 三角形尖角指向的一侧是高分区\n"
        << "  圆与方块颜色不同  -> 三角形尖角指向的反侧是高分区\n"
        << "\n"
        << "退出码:\n"
        << "  0 成功   2 参数错误   3 打不开图像/摄像头\n"
        << "  5 没识别出结果   6 写文件失败\n";
}

bool parseArgs(int argc, char** argv, Options& options, string& error)
{
    for (int i = 1; i < argc; ++i)
    {
        string arg = argv[i];
        string inlineValue;
        bool hasInlineValue = false;

        const size_t equal = arg.find('=');

        if (arg.rfind("--", 0) == 0 && equal != string::npos)
        {
            inlineValue = arg.substr(equal + 1);
            arg = arg.substr(0, equal);
            hasInlineValue = true;
        }

        auto takeValue = [&](string& value) -> bool
        {
            if (hasInlineValue)
            {
                value = inlineValue;
                return true;
            }

            if (i + 1 >= argc)
            {
                return false;
            }

            value = argv[++i];
            return true;
        };

        if (arg == "-h" || arg == "--help")
        {
            options.help = true;
        }
        else if (arg == "--image")
        {
            if (!takeValue(options.value))
            {
                error = "--image 需要一个图片路径";
                return false;
            }
            options.source = Options::Source::Image;
        }
        else if (arg == "--camera")
        {
            string value;

            if (!takeValue(value))
            {
                error = "--camera 需要一个编号";
                return false;
            }

            try
            {
                options.cameraIndex = stoi(value);
            }
            catch (const exception&)
            {
                error = "--camera 的取值不是整数：" + value;
                return false;
            }

            options.source = Options::Source::Camera;
        }
        else if (arg == "--out")
        {
            if (!takeValue(options.outDir))
            {
                error = "--out 需要一个目录路径";
                return false;
            }
        }
        else if (arg == "--votes")
        {
            string value;

            if (!takeValue(value))
            {
                error = "--votes 需要一个数字";
                return false;
            }

            try
            {
                options.votes = stoi(value);
            }
            catch (const exception&)
            {
                error = "--votes 的取值不是整数：" + value;
                return false;
            }

            options.votes = std::max(1, std::min(5, options.votes));
        }
        else if (arg == "--auto")
        {
            options.autoSave = true;
        }
        else if (arg == "--fast")
        {
            options.fast = true;
        }
        else if (arg == "--serial")
        {
            if (!takeValue(options.serialPort))
            {
                error = "--serial 需要一个串口设备路径";
                return false;
            }
        }
        else if (arg == "--baud")
        {
            string value;

            if (!takeValue(value))
            {
                error = "--baud 需要一个数字";
                return false;
            }

            try
            {
                options.serialBaud = stoi(value);
            }
            catch (const exception&)
            {
                error = "--baud 的取值不是整数：" + value;
                return false;
            }
        }
        else if (arg == "--fire")
        {
            options.fireAfterSend = true;
        }
        else if (arg == "--no-window")
        {
            options.showWindow = false;
        }
        else if (arg == "--debug")
        {
            options.debug = true;
        }
        else
        {
            error = "未知参数：" + arg;
            return false;
        }
    }

    return true;
}

// ---------------------------------------------------------------------------
// 公共：输出目录、保存、打印
// ---------------------------------------------------------------------------

bool prepareOutput(const Options& options, fs::path& outDir)
{
    error_code ec;
    fs::create_directories(options.outDir, ec);

    if (ec)
    {
        cerr << "无法创建输出目录 " << options.outDir << "：" << ec.message() << endl;
        return false;
    }

    outDir = fs::path(options.outDir);
    return true;
}

bool saveAll(const Options& options, const fs::path& outDir,
             const cv::Mat& canvas, const TagResult& result)
{
    Visualizer visualizer;

    if (!visualizer.saveImage(canvas, (outDir / "result.png").string()))
    {
        return false;
    }

    if (!visualizer.saveText(result, (outDir / "result.txt").string()))
    {
        return false;
    }

    if (options.debug && !result.warped.empty())
    {
        visualizer.saveImage(result.warped, (outDir / "tag_warped.png").string());
    }

    return true;
}

void printResult(const TagResult& result)
{
    cout << "\n---------- 识别结果 ----------\n";
    cout << highScoreText(result.highSide) << "\n";
    cout << "  标签     : " << (result.tagFound ? "已找到" : "未找到") << "\n";
    cout << "  尖角朝向 : " << sideName(result.tipSide)
         << "（把握 " << result.tipMargin << "）\n";
    cout << "  圆形颜色 : " << colorName(result.circleColor) << "\n";
    cout << "  方块颜色 : " << colorName(result.squareColor) << "\n";
    cout << "  颜色关系 : " << (result.sameColor ? "相同（取尖角侧）" : "不同（取反侧）") << "\n";
    cout << "  说明     : " << result.reason << "\n";
    cout << "------------------------------\n\n";
}

// 开机探针：先发 P，确认单片机固件在线
static bool probeMcu(SerialLink& link)
{
    for (int attempt = 0; attempt < 3; ++attempt)
    {
        if (!link.sendLine("P"))
        {
            return false;
        }

        if (link.readLine(400) == "PONG")
        {
            return true;
        }
    }

    return false;
}

// 发一条命令并等 ACK，最多重试 3 次（掉线/单片机复位都能救回来）
static bool sendWithAck(SerialLink& link, const string& command, const string& expectPrefix)
{
    for (int attempt = 1; attempt <= 3; ++attempt)
    {
        if (!link.sendLine(command))
        {
            cerr << "串口写入失败。" << endl;
            return false;
        }

        const string reply = link.readLine(600);

        if (reply.rfind(expectPrefix, 0) == 0)
        {
            return true;
        }

        cout << "第 " << attempt << " 次没收到 " << expectPrefix
             << (reply.empty() ? "（超时）" : "（收到：" + reply + "）")
             << "，重发……" << endl;

        this_thread::sleep_for(chrono::milliseconds(80));
    }

    return false;
}

// 把识别结果发给单片机：L = 左转，R = 右转；加 --fire 再发 F
void sendToMcu(const Options& options, const TagResult& result)
{
    if (options.serialPort.empty() || !result.decided)
    {
        return;
    }

    SerialLink link;

    if (!link.open(options.serialPort, options.serialBaud))
    {
        cerr << "串口没打开，跳过发送（识别结果本身不受影响）。" << endl;
        return;
    }

    link.flush();

    if (!probeMcu(link))
    {
        cerr << "\n没有收到单片机的 PONG —— 说明“PC → 板子 / 板子 → PC”有一条不通。按顺序查：\n"
             << "  1) 板子上跑的是不是 RM_Link_Test 固件？上电后 PC13 LED 应约 2 秒闪一下。\n"
             << "     不闪 = 没烧进去 / BOOT0 还在 1：拨回 0 再按一次 RESET。\n"
             << "     烧录命令：stm32flash -w RM_Link_Test.hex -v -g 0x0 /dev/ttyUSB0（BOOT0=1 时）\n"
             << "  2) 先看板子有没有主动打招呼：timeout 2 cat /dev/ttyUSB0\n"
             << "     能出 BOOT:RM_LINK → 板子→PC 通，问题在 PC→板子（TXD 到 PA10）\n"
             << "     出不来       → RXD（PA9）没接好 / 接反 / 没共地\n"
             << "  3) CH340 与板子必须交叉并共地：TXD→PA10、RXD→PA9、GND→GND\n"
             << "  4) CH340 跳线 3.3V，波特率 115200，串口没被 minicom 等占用\n"
             << "  5) 仍不行就把 TXD/RXD 对调试一次（有的模块丝印是模块视角）\n" << endl;
        return;
    }

    cout << "单片机在线（PONG）" << endl;

    const bool isLeft = (result.highSide == Side::Left);
    const string command = isLeft ? "L" : "R";

    if (!sendWithAck(link, command, "ACK:"))
    {
        cerr << "单片机在线但不认命令：确认板上固件是 RM_Link_Test（应回 ACK:L / ACK:R）。" << endl;
        return;
    }

    cout << "已发送串口命令：" << command
         << "（单片机 LED " << (isLeft ? "闪 1 下" : "闪 2 下") << " 表示"
         << (isLeft ? "左转" : "右转") << "）" << endl;

    if (options.fireAfterSend)
    {
        // 给单片机留出转向时间，再发开火
        this_thread::sleep_for(chrono::milliseconds(300));

        if (sendWithAck(link, "F", "ACK:"))
        {
            cout << "已发送开火命令：F（连打 5 发）" << endl;
        }
        else
        {
            cerr << "开火命令没有应答，检查单片机是否处于 ESTOP（发 S 后需要重新发 L/R）。" << endl;
        }
    }
}

// 锁定时在右上角加个提示框
void drawLockHint(cv::Mat& canvas, const TagResult& result)
{
    rectangle(canvas, cv::Rect(0, 0, canvas.cols - 1, canvas.rows - 1), cv::Scalar(0, 255, 0), 4);

    const string hint = "LOCKED  " + highScoreText(result.highSide) + "   SPACE=save  ESC=live";

    int baseline = 0;
    const cv::Size size = cv::getTextSize(hint, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &baseline);

    const cv::Rect box(canvas.cols - size.width - 30, 10, size.width + 20, size.height + 18);
    rectangle(canvas, box, cv::Scalar(0, 0, 0), cv::FILLED);
    rectangle(canvas, box, cv::Scalar(0, 255, 0), 1);

    cv::putText(canvas, hint, cv::Point(box.x + 10, box.y + size.height + 5),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
}

// ---------------------------------------------------------------------------
// 图片模式
// ---------------------------------------------------------------------------

int runImage(const Options& options)
{
    ImageSource source;
    cv::Mat image;

    if (!source.loadImage(options.value, image))
    {
        return 3;
    }

    fs::path outDir;
    if (!prepareOutput(options, outDir))
    {
        return 6;
    }

    TagDetector::Params params;

    if (options.fast)
    {
        params.analysisMaxSide = 640;
    }

    TagDetector detector(params);
    const TagResult result = detector.detect(image);

    Visualizer visualizer;
    const cv::Mat canvas = visualizer.drawResult(image, result);

    printResult(result);

    if (!saveAll(options, outDir, canvas, result))
    {
        return 6;
    }

    sendToMcu(options, result);

    cout << "已输出：" << (outDir / "result.png").string()
         << " / " << (outDir / "result.txt").string() << "\n";

    if (options.showWindow)
    {
        try
        {
            cv::imshow("RM HighScore", canvas);
            cout << "按任意键关闭窗口……\n";
            cv::waitKey(0);
            cv::destroyAllWindows();
        }
        catch (const cv::Exception& error)
        {
            cerr << "无法打开显示窗口（可能没有图形界面）：" << error.what() << "\n";
        }
    }

    return result.decided ? 0 : 5;
}

// ---------------------------------------------------------------------------
// 摄像头模式
//
// 单帧偶尔会被反光/抖动带偏，所以这里做"多帧投票"：
// 最近 5 帧里有 N 帧给出同一个结论才锁定，锁定后画面冻结，等裁判确认。
// ---------------------------------------------------------------------------

int runCamera(const Options& options)
{
    ImageSource source;

    if (!source.openCamera(options.cameraIndex))
    {
        return 3;
    }

    fs::path outDir;
    if (!prepareOutput(options, outDir))
    {
        return 6;
    }

    TagDetector::Params params;

    if (options.fast)
    {
        params.analysisMaxSide = 640;
    }

    TagDetector detector(params);
    Visualizer visualizer;

    cout << "把视觉标签（黑色方块那张纸）对准摄像头：" << endl;
    cout << "  连续 " << options.votes << " 帧判定一致后自动锁定" << endl;
    cout << "  [空格] 保存结果并退出    [ESC] 解除锁定回到实时画面    [q] 退出" << endl;

    cv::Mat frame;
    cv::Mat liveCanvas;

    std::deque<Side> history;
    bool locked = false;
    cv::Mat lockedCanvas;
    TagResult lockedResult;

    double fps = 0.0;
    int frameCount = 0;
    auto lastTick = chrono::steady_clock::now();

    while (true)
    {
        if (!locked)
        {
            if (!source.read(frame))
            {
                cerr << "读取摄像头画面失败！" << endl;
                break;
            }

            const TagResult result = detector.detect(frame);
            liveCanvas = visualizer.drawResult(frame, result);

            // 记录这一帧的结论
            history.push_back(result.decided ? result.highSide : Side::Unknown);

            while (history.size() > 5)
            {
                history.pop_front();
            }

            const int leftVotes = static_cast<int>(std::count(history.begin(), history.end(), Side::Left));
            const int rightVotes = static_cast<int>(std::count(history.begin(), history.end(), Side::Right));

            if ((leftVotes >= options.votes && leftVotes > rightVotes) ||
                (rightVotes >= options.votes && rightVotes > leftVotes))
            {
                locked = true;

                lockedResult = result;
                lockedResult.highSide = (leftVotes > rightVotes) ? Side::Left : Side::Right;
                lockedResult.decided = true;
                lockedResult.sameColor = result.sameColor;

                lockedCanvas = liveCanvas.clone();
                drawLockHint(lockedCanvas, lockedResult);

                cout << "\n" << highScoreText(lockedResult.highSide)
                     << "  (投票 " << max(leftVotes, rightVotes) << "/5，已锁定)" << endl;

                if (options.autoSave)
                {
                    printResult(lockedResult);

                    if (!saveAll(options, outDir, lockedCanvas, lockedResult))
                    {
                        source.release();
                        cv::destroyAllWindows();
                        return 6;
                    }

                    sendToMcu(options, lockedResult);

                    cout << "已保存：" << (outDir / "result.png").string()
                         << " / " << (outDir / "result.txt").string() << "\n";

                    source.release();
                    cv::destroyAllWindows();
                    return 0;
                }
            }
            else
            {
                // 帧率统计
                ++frameCount;

                const auto now = chrono::steady_clock::now();
                const double seconds = chrono::duration<double>(now - lastTick).count();

                if (seconds >= 0.5)
                {
                    fps = static_cast<double>(frameCount) / seconds;
                    frameCount = 0;
                    lastTick = now;
                }

                ostringstream info;
                info << "fps " << static_cast<int>(fps + 0.5)
                     << "   votes " << max(leftVotes, rightVotes) << "/5";

                cv::putText(liveCanvas, info.str(), cv::Point(20, liveCanvas.rows - 64),
                            cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2, cv::LINE_AA);
            }

            cv::imshow("RM HighScore", liveCanvas);
        }
        else
        {
            cv::imshow("RM HighScore", lockedCanvas);
        }

        const int key = cv::waitKey(1) & 0xFF;

        if (key == 'q')
        {
            break;
        }

        if (key == 27)
        {
            if (locked)
            {
                // 解除锁定，回到实时画面重新识别
                locked = false;
                history.clear();
                cout << "已解除锁定，继续实时识别。" << endl;
                continue;
            }

            break;
        }

        if ((key == ' ' || key == 13) && locked)
        {
            printResult(lockedResult);

            if (!saveAll(options, outDir, lockedCanvas, lockedResult))
            {
                source.release();
                cv::destroyAllWindows();
                return 6;
            }

            sendToMcu(options, lockedResult);

            cout << "已保存：" << (outDir / "result.png").string()
                 << " / " << (outDir / "result.txt").string() << "\n";

            source.release();
            cv::destroyAllWindows();
            return 0;
        }
    }

    source.release();
    cv::destroyAllWindows();

    return 3;
}

// ---------------------------------------------------------------------------

int main(int argc, char** argv)
{
    const auto programStart = chrono::steady_clock::now();

    Options options;
    string argError;

    if (!parseArgs(argc, argv, options, argError))
    {
        cerr << "参数错误：" << argError << "\n\n";
        printUsage(argv[0]);
        return 2;
    }

    if (options.help)
    {
        printUsage(argv[0]);
        return 0;
    }

    cout << "========================================================\n";
    cout << "   RM 识别高分区   |   high score: left / right\n";
    cout << "========================================================\n";

    const int code = (options.source == Options::Source::Image)
        ? runImage(options)
        : runCamera(options);

    const double totalMs = chrono::duration<double, std::milli>(
        chrono::steady_clock::now() - programStart).count();

    cout << "\n总耗时：" << fixed << setprecision(1) << totalMs << " ms\n";

    return code;
}

