// NonMaxSup.hpp --- 
// 
// Filename: NonMaxSup.hpp
// Description: 
// Author: Yannick Verdie, Kwang Moo Yi, Alberto Crivella
// Maintainer: Yannick Verdie, Kwang Moo Yi
// Created: Tue Mar  3 17:51:23 2015 (+0100)
// Version: 0.5a
// Package-Requires: ()
// Last-Updated: Thu May 28 12:53:23 2015 (+0200)
//           By: Kwang
//     Update #: 8
// URL: 
// Doc URL: 
// Keywords: 
// Compatibility: 
// 
// 

// Commentary: 
// 
// 
// 
// 

// Change Log:
// 
// 
// 
// 
// Copyright (C), EPFL Computer Vision Lab.
// 
// 

// Code:


#ifndef _NON_MAX_SUP_HPP_
#define _NON_MAX_SUP_HPP_


#include <iostream>
#include <fstream>
#include <string>

 #include <opencv2/opencv.hpp>

#ifdef __ANDROID__
#include <android/log.h>
#define  LOG_TAG    "TILDE"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

inline float stof(const string& f){return atof(f.c_str()); };
inline int stoi(const string& f){return atoi(f.c_str()); };
#else
#define  LOGD(...)  printf(__VA_ARGS__);//std::cout<<__VA_ARGS__<<std::endl;
#define  LOGE(...)  printf(__VA_ARGS__);//throw std::runtime_error(__VA_ARGS__);
#endif


using namespace std;
using namespace cv;

vector<Point3f> NonMaxSup(const Mat & response);
vector<KeyPoint> NonMaxSup_resize_format(const Mat &response, const float& resizeRatio, const float &scaleKeypoint, const float & orientationKeypoint, const bool sortMe = false);



#endif // _NON_MAX_SUP_HPP_

// 
// NonMaxSup.hpp ends here
