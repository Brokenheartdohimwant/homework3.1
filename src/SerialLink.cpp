#include "SerialLink.hpp"

#include <cerrno>
#include <chrono>
#include <cstring>
#include <iostream>
#include <thread>

#if defined(__linux__) || defined(__APPLE__)

#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

namespace
{
    speed_t toSpeed(int baud, bool& ok)
    {
        ok = true;

        switch (baud)
        {
            case 9600:   return B9600;
            case 19200:  return B19200;
            case 38400:  return B38400;
            case 57600:  return B57600;
            case 115200: return B115200;
            case 230400: return B230400;
            default:     ok = false; return B115200;
        }
    }
}

SerialLink::~SerialLink()
{
    close();
}

bool SerialLink::open(const std::string& port, int baud)
{
    close();

    fd_ = ::open(port.c_str(), O_RDWR | O_NOCTTY | O_NONBLOCK);

    if (fd_ < 0)
    {
        std::cerr << "打开串口失败：" << port << "（" << std::strerror(errno) << "）" << std::endl;
        std::cerr << "权限问题可以执行：sudo usermod -aG dialout $USER 然后重新登录" << std::endl;
        return false;
    }

    termios tty {};
    bool speedOk = true;

    if (tcgetattr(fd_, &tty) != 0)
    {
        std::cerr << "tcgetattr 失败：" << std::strerror(errno) << std::endl;
        close();
        return false;
    }

    cfmakeraw(&tty);

    const speed_t speed = toSpeed(baud, speedOk);

    if (!speedOk)
    {
        std::cerr << "不支持的波特率：" << baud << "，使用 115200" << std::endl;
    }

    cfsetispeed(&tty, speed);
    cfsetospeed(&tty, speed);

    tty.c_cflag |= (CLOCAL | CREAD);
    tty.c_cflag &= ~PARENB;
    tty.c_cflag &= ~CSTOPB;
    tty.c_cflag &= ~CSIZE;
    tty.c_cflag |= CS8;

    tty.c_cc[VMIN]  = 0;
    tty.c_cc[VTIME] = 1;

    if (tcsetattr(fd_, TCSANOW, &tty) != 0)
    {
        std::cerr << "tcsetattr 失败：" << std::strerror(errno) << std::endl;
        close();
        return false;
    }

    tcflush(fd_, TCIOFLUSH);

    return true;
}

bool SerialLink::sendLine(const std::string& line)
{
    if (fd_ < 0)
    {
        return false;
    }

    const std::string data = line + "\n";

    std::size_t sent = 0;

    while (sent < data.size())
    {
        const ssize_t n = ::write(fd_, data.data() + sent, data.size() - sent);

        if (n > 0)
        {
            sent += static_cast<std::size_t>(n);
            continue;
        }

        if (n < 0 && (errno == EINTR))
        {
            continue;
        }

        if (n < 0 && (errno == EAGAIN || errno == EWOULDBLOCK))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        std::cerr << "串口写入失败：" << std::strerror(errno) << std::endl;
        return false;
    }

    tcdrain(fd_);

    return true;
}

std::string SerialLink::readLine(int timeoutMs)
{
    if (fd_ < 0)
    {
        return std::string();
    }

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeoutMs);

    while (true)
    {
        const std::size_t newline = buffer_.find('\n');

        if (newline != std::string::npos)
        {
            std::string line = buffer_.substr(0, newline);
            buffer_.erase(0, newline + 1);

            if (!line.empty() && line.back() == '\r')
            {
                line.pop_back();
            }

            return line;
        }

        char chunk[128];
        const ssize_t n = ::read(fd_, chunk, sizeof(chunk));

        if (n > 0)
        {
            buffer_.append(chunk, static_cast<std::size_t>(n));
            continue;
        }

        if (std::chrono::steady_clock::now() > deadline)
        {
            return std::string();
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
}

void SerialLink::close()
{
    if (fd_ >= 0)
    {
        ::close(fd_);
        fd_ = -1;
    }

    buffer_.clear();
}

#else   // 非 POSIX 平台：编译通过但不工作

SerialLink::~SerialLink() {}

bool SerialLink::open(const std::string& port, int)
{
    std::cerr << "串口只支持 Linux/macOS（当前平台不可用）：" << port << std::endl;
    return false;
}

bool SerialLink::sendLine(const std::string&) { return false; }

std::string SerialLink::readLine(int) { return std::string(); }

void SerialLink::close() {}

#endif