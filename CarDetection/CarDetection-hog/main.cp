//
//  main.cpp
//  imagedisplay
//
//  Created by CSQS on 14-9-28.
//  Copyright (c) 2014年 user. All rights reserved.
//

#include "cv.h"
#include "highgui.h"
#include <ml.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using namespace std;
using namespace cv;

//图片尺寸以及相关参数

//使用数据集图像大小 100*40

//hog参数 依次是 window block stride cell bin
//参考参数 （64，64）（16，16）（8，8）（8，8） 9
//window向量个数 1764
//使用参数 （100，40）（50，20）（25，10）（25，10）9
//window向量个数 {[(50/25) * (20/10)] * 9} * {[(100-50)/25+1]*[(40-20)/10+1]}= 36*9=324

const int image_width = 100;
const int image_height = 40;
const int win_width = 100;
const int win_height = 40;
const int block_width = 50;
const int block_height = 20;
const int stride_width = 25;
const int stride_height = 10;
const int cell_width = 25;
const int cell_height = 10;

int hog_dsize = 324;

//hog.detectMultiScale 未优化参数设置

//训练数量样本
const int image_neg = 499;
const int image_pos = 549;

//hardexample个数
int hard_example = 0;

//找到hardexample
void FindHardexample();
//进行车辆检测
void Cardetect();

class Mysvm: public CvSVM
{
    public:
        int get_alpha_count()
        {
            return this->sv_total;
        }
        int get_sv_dim()
        {
            return this->var_all;
        }
        int get_sv_count()
        {
            return this->decision_func->sv_count;
        }
        double* get_alpha()
        {
            return this->decision_func->alpha;
        }
        float** get_sv()
        {
            return this->sv;
        }
        float get_rho()
        {
            return this->decision_func->rho;
        }
};

//找出hardexample
void FindHardexample()
{
    char filename[256];
    char address[256] = "/Users/user/Desktop/MY 320/CarDetection/CarData/TrainHardImages";
    char fullname[256];
    
    //输入svm模型形成hog检测子
    vector<float> x;
    ifstream fileIn("/Users/user/Desktop/MY 320/CarDetection/result/cardetectSVM.txt", ios::in);
    float val = 0.0f;
    
    while(!fileIn.eof()){
        fileIn>>val;
        x.push_back(val);
    }
    fileIn.close();
    
    vector<cv::Rect>  found;
    cv::HOGDescriptor hog(cvSize(win_width,win_height), cvSize(block_width,block_height), cvSize(stride_width,stride_height), cvSize(cell_width,cell_height),9);
    hog.setSVMDetector(x);
    
    //检测负样本找出hardexample
    int image_id = 0;
    IplImage* src;
    sprintf(filename,"/Users/user/Desktop/MY 320/CarDetection/CarData/TrainImages/neg-%d.pgm",image_id);
    src = cvLoadImage(filename);
    
    while(src){
        //进行多尺度检测
        hog.detectMultiScale(src, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);
        if (found.size() > 0){
            for (int i = 0; i < found.size(); i++){
                //将错误的检测区域裁剪并缩放成样本尺寸
                Rect hardrec = found[i];
                //将图片外坐标处理进入图片内 考虑坐标为负数或超过图片尺寸的情况
                if(hardrec.x < 0) hardrec.x = 0;
                if(hardrec.y < 0) hardrec.y = 0;
                if(hardrec.x + hardrec.width > image_width) hardrec.width = image_width - hardrec.x;
                if(hardrec.y + hardrec.height > image_height) hardrec.height = image_height - hardrec.y;
                //缩放
                IplImage* hardExampleImg;
                IplImage* resizeImg = cvCreateImage(cvSize(image_width,image_height), src->depth, src->nChannels);
                
                cvSetImageROI(src,hardrec);
                hardExampleImg = cvCreateImage(cvSize(hardrec.width, hardrec.height), src->depth, src->nChannels );
                cvResize(hardExampleImg, resizeImg, CV_INTER_LINEAR);
                cvResetImageROI(src);
                //保存
                hard_example++;
                sprintf(fullname,"%s/hardexample-%d.pgm",address,hard_example);//生成hard example图片的文件名
                cout<<fullname<<endl;
                cvSaveImage(fullname, resizeImg);
            }
        }
        image_id ++;
        sprintf(filename,"/Users/user/Desktop/MY 320/CarDetection//CarData/TrainImages/neg-%d.pgm",image_id);
        src = cvLoadImage(filename);
    }
}

void Cardetect()
{
    //提取样本的hog特征
    char filename[256];
    int image_id = 0;
    int image_total = image_neg + image_pos + hard_example + 1;
    int n = 0;
    
    sprintf(filename,"/Users/user/Desktop/MY 320/CarDetection/CarData/TrainImages/neg-%d.pgm",image_id);
    IplImage* src = cvLoadImage(filename);
    
    CvMat *data_mat, *res_mat;
    data_mat = cvCreateMat( image_total,hog_dsize, CV_32FC1);
    cvSetZero( data_mat );
    res_mat = cvCreateMat( image_total, 1 , CV_32FC1);
    cvSetZero( res_mat);
    
    
    //负样本
    image_id = 0;
    sprintf(filename,"/Users/user/Desktop/MY 320/CarDetection/CarData/TrainImages/neg-%d.pgm",image_id);
    src = cvLoadImage(filename);
    
    
    while(src){
        cout<<" process neg-"<<image_id<<endl;
        
        HOGDescriptor *hog = new HOGDescriptor(cvSize(win_width,win_height), cvSize(block_width,block_height), cvSize(stride_width,stride_height), cvSize(cell_width,cell_height),9);
        
        vector<float>descriptors;
        hog->compute(src, descriptors, Size(1,1), Size(0,0));
        n = 0;
        for(vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++){
            cvmSet(data_mat, image_id, n, *iter);
            n++;
        }
        cvmSet(res_mat, image_id, 0, 0);
        cout<< " end processing "<<endl;
        
        image_id ++;
        sprintf(filename,"/Users/user/Desktop/MY 320/CarDetection/CarData/TrainImages/neg-%d.pgm",image_id);
        src = cvLoadImage(filename);
    }
    
    
    //正样本
    image_id = 0;
    sprintf(filename,"/Users/user/Desktop/MY 320/CarDetection/CarData/TrainImages/pos-%d.pgm",image_id);
    src = cvLoadImage(filename);
    
    
    while(src){
        cout<<" process pos-"<<image_id<<endl;
        
        HOGDescriptor *hog = new HOGDescriptor(cvSize(win_width,win_height), cvSize(block_width,block_height), cvSize(stride_width,stride_height), cvSize(cell_width,cell_height),9);
        
        vector<float>descriptors;
        hog->compute(src, descriptors, Size(1,1), Size(0,0));
        cout<< "HOG dims: "<< descriptors.size()<<endl;
        n = 0;
        for(vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++){
            cvmSet(data_mat, image_id + image_neg, n, *iter);
            n++;
        }
        cvmSet(res_mat, image_id + image_neg, 0, 1);
        cout<< " end processing "<<endl;
        
        image_id ++;
        sprintf(filename,"/Users/user/Desktop/MY 320/CarDetection/CarData/TrainImages/pos-%d.pgm",image_id);
        src = cvLoadImage(filename);
    }
    
    //hardexample
    image_id = 0;
    sprintf(filename,"/Users/user/Desktop/MY 320/CarDetection/CarData/TrainHardImages/hardexample-%d.pgm",image_id);
    src = cvLoadImage(filename);
    
    while(src){
        cout<<" process hardexample-"<<image_id<<endl;
        
        HOGDescriptor *hog = new HOGDescriptor(cvSize(win_width,win_height), cvSize(block_width,block_height), cvSize(stride_width,stride_height), cvSize(cell_width,cell_height),9);
        
        vector<float>descriptors;
        hog->compute(src, descriptors, Size(1,1), Size(0,0));
        cout<< "HOG dims: "<< descriptors.size()<<endl;
        n = 0;
        for(vector<float>::iterator iter = descriptors.begin(); iter != descriptors.end(); iter++){
            cvmSet(data_mat, image_id + image_neg, n, *iter);
            n++;
        }
        cvmSet(res_mat, image_id + image_neg, 0, 0);
        cout<< " end processing "<<endl;
        
        image_id ++;
        sprintf(filename,"/Users/user/Desktop/MY 320/CarDetection/CarData/TrainHardImages/hardexample-%d.pgm",image_id);
        src = cvLoadImage(filename);
    }
    
    //进行SVM学习
    
    Mysvm svm;
    CvSVMParams param;
    CvTermCriteria criteria;
    criteria = cvTermCriteria( CV_TERMCRIT_EPS, 1000, FLT_EPSILON);
    param = CvSVMParams( CvSVM::C_SVC, CvSVM::LINEAR, 10.0, 0.09, 1.0, 10.0, 0.5, 1.0, NULL,criteria);
    char classifierSavePath[256] = "/Users/user/Desktop/MY 320/CarDetection/result/cardetectSVM.txt";
    
    //得到分类器
    svm.train(data_mat,res_mat,NULL,NULL,param);
    svm.save(classifierSavePath);
    
    int supportVectorSize = svm.get_support_vector_count();
    
    CvMat *sv,*alp,*re;//所有样本特征向量
    sv  = cvCreateMat(supportVectorSize , hog_dsize, CV_32FC1);
    alp = cvCreateMat(1 , supportVectorSize, CV_32FC1);
    re  = cvCreateMat(1 , hog_dsize, CV_32FC1);
    
    cvSetZero(sv);
    cvSetZero(re);
    
    //将分类器处理成检测子
    for(int i=0; i<supportVectorSize; i++){
        memcpy( (float*)(sv->data.fl+i*hog_dsize), svm.get_support_vector(i), hog_dsize*sizeof(float));
    }
    
    double* alphaArr = svm.get_alpha();
    
    for(int i=0; i<supportVectorSize; i++){
        alp->data.fl[i] = alphaArr[i];
    }
    cvMatMul(alp, sv, re);
    
    for (int i=0; i<hog_dsize; i++){
        re->data.fl[i] *= -1;
    }
    
    FILE* fp = fopen("/Users/user/Desktop/MY 320/CarDetection/result/cardetectSVM.txt","wb");
    if( NULL == fp ){
        cout<<"/Users/user/Desktop/MY 320/CarDetection/result/cardetectSVM.txt 打开失败"<<endl;
    }
    for (int i=0; i<hog_dsize; i++){
        fprintf(fp,"%f \n",re->data.fl[i]);
    }
    float rho = svm.get_rho();
    fprintf(fp, "%f", rho);
    cout<<"/Users/user/Desktop/MY 320/CarDetection/result/cardetectSVM.txt 保存完毕"<<endl;
    fclose(fp);
    
    //检测
    vector<float> x;
    ifstream fileIn("/Users/user/Desktop/MY 320/CarDetection/result/cardetectSVM.txt", ios::in);
    float val = 0.0f;
    while(!fileIn.eof()){
        fileIn>>val;
        x.push_back(val);
    }
    fileIn.close();
    
    vector<cv::Rect>  found;
    ofstream file("/Users/user/Desktop/MY 320/CarDetection/result/locations.txt", ios::out);
    
    cv::HOGDescriptor hog(cvSize(win_width,win_height), cvSize(block_width,block_height), cvSize(stride_width,stride_height), cvSize(cell_width,cell_height),9);
    hog.setSVMDetector(x);
    
    image_id = 0;
    sprintf(filename,"/Users/user/Desktop/MY 320/CarDetection/CarData/TestImages/test-%d.pgm",image_id);
    src = cvLoadImage(filename);
    
    while(src){
        cout<<image_id<<": ";
        file<<image_id<<": ";
        hog.detectMultiScale(src, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);
        if (found.size() > 0){
            for (int i = 0; i < found.size(); i++){
                CvRect tempRect = cvRect(found[i].x, found[i].y, found[i].width, found[i].height);
                cvRectangle(src, cvPoint(tempRect.x,tempRect.y), cvPoint(tempRect.x+tempRect.width,tempRect.y+tempRect.height),CV_RGB(255,0,0), 2);
                //cout<<"("<<found[i].x<<","<<found[i].y<<") ";
                //file<<"("<<found[i].x<<","<<found[i].y<<") ";
                cout<<"("<<found[i].y<<","<<found[i].x<<") ";
                file<<"("<<found[i].y<<","<<found[i].x<<") ";

            }
        }
        cout<<endl;
        file<<endl;
        image_id ++;
        sprintf(filename,"/Users/user/Desktop/MY 320/CarDetection/CarData/TestImages/test-%d.pgm",image_id);
        src = cvLoadImage(filename);
    }
    file.close();
}


int main(int argc, const char * argv[])
{
    int inter = 10;
    
    //迭代多次提高准确率
    for(int i = 0; i < inter; i++){
        if(!hard_example) Cardetect();
        FindHardexample();
        Cardetect();
    }
    
    return 0;
}



