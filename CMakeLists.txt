cmake_minimum_required(VERSION 3.16)
project(mlp_desktop_integration LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(mlp_inference
    cpp/main.cpp
    cpp/MainWindow.cpp
    cpp/CsvReader.cpp
)

target_include_directories(mlp_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/cpp
)
