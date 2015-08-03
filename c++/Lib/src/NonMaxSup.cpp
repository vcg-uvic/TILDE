// NonMaxSup.cpp --- 
// 
// Filename: NonMaxSup.cpp
// Description: 
// Author: Yannick Verdie, Kwang Moo Yi
// Maintainer: Yannick Verdie, Kwang Moo Yi
// Created: Tue Mar  3 17:48:14 2015 (+0100)
// Version: 0.5a
// Package-Requires: ()
// Last-Updated: Tue Jun 16 17:09:37 2015 (+0200)
//           By: Kwang
//     Update #: 6
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
#include "NonMaxSup.hpp"

vector<Point3f> 
NonMaxSup(const Mat & response)
{
    // stupid non-max suppression without any fancy tricks
    vector<Point3f> res;
    for(int i=1; i<response.rows-1; ++i){
        for(int j=1; j<response.cols-1; ++j)
        {
            bool bMax = true;

            for(int ii=-1; ii <= +1; ++ii)
            for(int jj=-1; jj <= +1; ++jj){
                if(ii==0 && jj==0)
                    continue;
                bMax &= response.at<float>(i,j) > response.at<float>(i+ii,j+jj);
            }

            if (bMax)
            {
                res.push_back(Point3f(j,i,response.at<float>(i,j)));
                //cout<<i<<" "<<j<<endl;
            }

        }            
    }

    return res;
}

vector<KeyPoint> NonMaxSup_resize_format(const Mat &response, const float& resizeRatio, const float &scaleKeypoint, const float & orientationKeypoint, const bool sortMe)
{
	if (response.type() != CV_32F)
        LOGE("Wrong type in NMS %d",response.type());
    // stupid non-max suppression without any fancy tricks
    vector<KeyPoint> res;


    for(int i=1; i<response.rows-1; ++i)
    {
        const float* pixelinprev = response.ptr<float>(i-1); //previous line
        const float* pixelin = response.ptr<float>(i); //current line
        const float* pixelinnext = response.ptr<float>(i+1); //next line

        for(int j=1; j<response.cols-1; ++j)
        {
            bool bMax = true;
            const float val = *pixelin;//response.at<float>(i,j);

            //for(int ii=-1; ii <= +1; ++ii)
            for(int jj=-1; jj <= +1 && bMax; ++jj)
            {
                if (*(pixelinprev+jj) >= val) bMax = false;
                if (*(pixelin+jj) >= val && jj != 0) bMax = false;
                if (*(pixelinnext+jj) >= val) bMax = false;
            }

            if (bMax)
            {
                res.push_back(KeyPoint(Point2f(j * resizeRatio, i * resizeRatio), scaleKeypoint,orientationKeypoint,val));
            }

            pixelin++;//next
            pixelinnext++;//next
            pixelinprev++;//next

        }            
    }

    if (sortMe) {
        std::sort(res.begin(), res.end(),
              [](const KeyPoint & a, const KeyPoint & b) {
              return a.response > b.response;}
        );
    }


    return res;
}



// NonMaxSup.cpp ends here
