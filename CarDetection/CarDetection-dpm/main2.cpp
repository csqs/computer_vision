//
//  main.cpp
//  opencv-dpm
//
//  Created by Mercedes on 14/10/19.
//
//  Copyright (c) 2014年 Mercedes. All rights reserved.
//

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <cstring>

#ifdef HAVE_CVCONFIG_H
#include <cvconfig.h>
#endif
#ifdef HAVE_TBB
#include "tbb/task_scheduler_init.h"
#endif

using namespace cv;

const char* model_filename = "/Users/mercedes/Desktop/MY 320/CarDetection-dpm/归档/model/car_final.xml";
//const char* image_filename = "/Users/mercedes/Desktop/MY 320/CarDetection-dpm/归档/images/test/5/test5-0000";
//const char* image_filename = "/Users/mercedes/Desktop/MY 320/CarDetection-dpm/归档/images/test/5/test5-000035.JPG";
const char* image_filename = "/Users/mercedes/Desktop/MY 320/CarDetection-dpm/归档/images/test/7/test5-000034.JPG";
//const char* image_filename = "/Users/mercedes/Desktop/MY 320/CarDetection-dpm/归档/images/test/5/test5-000037.JPG";
int   tbbNumThreads = 4;
/*
void detect_and_draw_objects( IplImage* image, CvLatentSvmDetector* detector, int numThreads = -1)
{
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* detections = 0;
    int i = 0;
    int64 start = 0, finish = 0;
#ifdef HAVE_TBB
    tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);
    if (numThreads > 0)
    {
        init.initialize(numThreads);
        printf("Number of threads %i\n", numThreads);
    }
    else
    {
        printf("Number of threads is not correct for TBB version");
        return;
    }
#endif
    
    start = cvGetTickCount();
    detections = cvLatentSvmDetectObjects(image, detector, storage, 0.5f, numThreads);
    finish = cvGetTickCount();
    printf("detection time = %.3f\n", (float)(finish - start) / (float)(cvGetTickFrequency() * 1000000.0));
    
#ifdef HAVE_TBB
    init.terminate();
#endif
    for( i = 0; i < detections->total; i++ )
    {
        CvObjectDetection detection = *(CvObjectDetection*)cvGetSeqElem( detections, i );
        CvRect bounding_box = detection.rect;
        cvRectangle( image, cvPoint(bounding_box.x, bounding_box.y),
                    cvPoint(bounding_box.x + bounding_box.width,
                            bounding_box.y + bounding_box.height),
                    CV_RGB(255,0,0), 3 );
    }
    cvReleaseMemStorage( &storage );
}
*/

void detect_and_draw_objects( IplImage* image, CvLatentSvmDetector* detector, int numThreads = -1)
{
    CvMemStorage* storage = cvCreateMemStorage(0);
    CvSeq* detections = 0;
    int i = 0;
    int j = 0;
    int judge = 0;
    int64 start = 0, finish = 0;
#ifdef HAVE_TBB
    tbb::task_scheduler_init init(tbb::task_scheduler_init::deferred);
    if (numThreads > 0)
    {
        init.initialize(numThreads);
        printf("Number of threads %i\n", numThreads);
    }
    else
    {
        printf("Number of threads is not correct for TBB version");
        return;
    }
#endif
    
    start = cvGetTickCount();
    detections = cvLatentSvmDetectObjects(image, detector, storage, 0.5f, numThreads);
    finish = cvGetTickCount();
    printf("detection time = %.3f\n", (float)(finish - start) / (float)(cvGetTickFrequency() * 1000000.0));
    
#ifdef HAVE_TBB
    init.terminate();
#endif
    for( i = 0; i < detections->total; i++ )
    {
        CvObjectDetection detection_i = *(CvObjectDetection*)cvGetSeqElem( detections, i );
        CvRect bounding_box_i = detection_i.rect;
        for ( j = 0; j < detections->total; j++ )
        {
            CvObjectDetection detection_j = *(CvObjectDetection*)cvGetSeqElem( detections, j );
            CvRect bounding_box_j = detection_j.rect;
            
            if (bounding_box_i.x < bounding_box_j.x && bounding_box_i.y < bounding_box_j.y && bounding_box_i.x + bounding_box_i.width > bounding_box_j.x + bounding_box_j.width && bounding_box_i.y + bounding_box_i.height > bounding_box_j.y + bounding_box_j.height)
            {
                judge = 1;
            }
        }
        if (!judge)
            cvRectangle( image, cvPoint(bounding_box_i.x, bounding_box_i.y),cvPoint(bounding_box_i.x + bounding_box_i.width,bounding_box_i.y + bounding_box_i.height),CV_RGB(255,0,0), 3 );
        judge = 0;
        
    }
    cvReleaseMemStorage( &storage );
}

int main(int argc, char* argv[])
{
    if (argc > 2)
    {
        image_filename = argv[1];
        model_filename = argv[2];
        if (argc > 3)
        {
            tbbNumThreads = atoi(argv[3]);
        }
    }
    
    char *num;

        IplImage* image = cvLoadImage(image_filename);
        if (!image)
        {
            printf( "Unable to load the image\n"
                   "Pass it as the first parameter: latentsvmdetect <path to cat.jpg> <path to cat.xml>\n" );
            return -1;
        }
            CvLatentSvmDetector* detector = cvLoadLatentSvmDetector(model_filename);
        if (!detector)
        {
            printf( "Unable to load the model\n"
                   "Pass it as the second parameter: latentsvmdetect <path to cat.jpg> <path to cat.xml>\n" );
            cvReleaseImage( &image );
            return -1;
        }
        detect_and_draw_objects( image, detector, tbbNumThreads );
        cvNamedWindow( "test", 0 );
        cvShowImage( "test", image );
        cvWaitKey(0);
        cvReleaseLatentSvmDetector( &detector );
        cvReleaseImage( &image );
        cvDestroyAllWindows();
        getchar();

    
    return 0;
}

