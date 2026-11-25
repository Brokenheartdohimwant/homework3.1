#ifndef SERIAL_LINK_HPP
#define SERIAL_LINK_HPP

#include <string>

// 上位机 -> 单片机 的串口链路（Linux / macOS，POSIX termios）
//
// 用法：
//   SerialLink link;
//   if (link.open("/dev/ttyUSB0", 115200)) { link.sendLine("L"); }
//
// 协议见 RM_Mortar_Firmware/docs/PROTOCOL.md
class SerialLink
{
public:
    SerialLink() = default;
    ~SerialLink();

    SerialLink(const SerialLink&) = delete;
    SerialLink& operator=(const SerialLink&) = delete;

    bool open(const std::string& port, int baud = 115200);
    bool isOpen() const { return fd_ >= 0; }

    // 发送一行（自动补 '\n'）
    bool sendLine(const std::string& line);

    // 读一行（不含 '\n'）；超时返回空串
    std::string readLine(int timeoutMs = 500);

    void close();

private:
    int fd_ = -1;
    std::string buffer_;
};

#endif