cmake_minimum_required(VERSION 3.10)

project(RM_HighScore LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV REQUIRED COMPONENTS
    core
    imgproc
    highgui
    videoio
    imgcodecs
)

add_executable(
    RM_HighScore
    src/main.cpp
    src/TagDetector.cpp
    src/ImageSource.cpp
    src/SerialLink.cpp
    src/Visualizer.cpp
)

target_include_directories(RM_HighScore PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(RM_HighScore PRIVATE
    ${OpenCV_LIBS}
)

# 源码里有中文提示信息，MSVC 需要显式声明 UTF-8
if(MSVC)
    target_compile_options(RM_HighScore PRIVATE /utf-8)
endif()
