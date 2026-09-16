# Verifies an auto-tuned template with the C++ edition of Dynamsoft Barcode Reader.
#
#   cmake -S . -B build -DDBR_SDK_ROOT=<folder containing Include/ and Dist/>
#   cmake --build build --config Release
#
# DBR_SDK_ROOT is the root of the C++ SDK or of the barcode-reader-c-cpp-samples
# checkout - anything with Include/DynamsoftCaptureVisionRouter.h in it.

cmake_minimum_required(VERSION 3.16)
project(dbr_verify CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED DBR_SDK_ROOT)
    message(FATAL_ERROR "Set DBR_SDK_ROOT to the SDK folder that holds Include/ and Dist/")
endif()

if(NOT EXISTS "${DBR_SDK_ROOT}/Include/DynamsoftCaptureVisionRouter.h")
    message(FATAL_ERROR "DynamsoftCaptureVisionRouter.h not found under ${DBR_SDK_ROOT}/Include")
endif()

if(WIN32)
    set(DBR_ARCH x64)
    if(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(DBR_ARCH x86)
    endif()
    set(DBR_LIB_DIR "${DBR_SDK_ROOT}/Dist/Lib/Windows/${DBR_ARCH}")
    set(DBR_LIBS
        "${DBR_LIB_DIR}/DynamsoftCore${DBR_ARCH}.lib"
        "${DBR_LIB_DIR}/DynamsoftLicense${DBR_ARCH}.lib"
        "${DBR_LIB_DIR}/DynamsoftCaptureVisionRouter${DBR_ARCH}.lib")
elseif(APPLE)
    set(DBR_LIB_DIR "${DBR_SDK_ROOT}/Dist/Lib/MacOS")
    set(DBR_LIBS
        "${DBR_LIB_DIR}/libDynamsoftCore.dylib"
        "${DBR_LIB_DIR}/libDynamsoftLicense.dylib"
        "${DBR_LIB_DIR}/libDynamsoftCaptureVisionRouter.dylib")
else()
    set(DBR_ARCH x64)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(DBR_ARCH ARM64)
    endif()
    set(DBR_LIB_DIR "${DBR_SDK_ROOT}/Dist/Lib/Linux/${DBR_ARCH}")
    set(DBR_LIBS
        "${DBR_LIB_DIR}/libDynamsoftCore.so"
        "${DBR_LIB_DIR}/libDynamsoftLicense.so"
        "${DBR_LIB_DIR}/libDynamsoftCaptureVisionRouter.so")
endif()

add_executable(dbr_verify main.cpp)
target_include_directories(dbr_verify PRIVATE "${DBR_SDK_ROOT}/Include")
target_link_libraries(dbr_verify PRIVATE ${DBR_LIBS})

if(UNIX)
    set_target_properties(dbr_verify PROPERTIES
        BUILD_RPATH "${DBR_LIB_DIR}" INSTALL_RPATH "${DBR_LIB_DIR}")
endif()

# The runtime needs every shared library plus the Models folder beside the exe.
if(WIN32)
    add_custom_command(TARGET dbr_verify POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${DBR_LIB_DIR}" "$<TARGET_FILE_DIR:dbr_verify>"
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${DBR_SDK_ROOT}/Dist/Models" "$<TARGET_FILE_DIR:dbr_verify>/Models"
        COMMENT "Copying DBR runtime next to dbr_verify")
endif()
