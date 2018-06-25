#include <iostream>
#include <deque>
#include <boost/thread/thread.hpp>
#include <boost/thread/shared_mutex.hpp>
#include <boost/bind.hpp>
#include <string>
#include <sstream>
#include <string.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <stdlib.h>
#include <math.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <caffe/caffe.hpp>
#include <opencv2/opencv.hpp>
#include "kcftracker.hpp"
#include "tracker.h"
#include "lane_detection.hpp"
#include "object_detection.hpp"
#include <algorithm>
#include "can_serial.hpp"

#define L_PI 3.14159265358979323846

using namespace caffe;
using namespace std;
using namespace cv;


//-----------------初始化传入可执行文件的参数,给定参数说明和默认值,详见run.sh--------------------
DEFINE_string(model_file, "",
    "The model file used to initialize network architecture (*.prototxt).");
DEFINE_string(weights_file, "",
    "The weights file used to initialize network parameter (*.caffemodel)");
DEFINE_int32(camera_id, 0,
    "The camera ID");
DEFINE_string(ipm_file, "",
    "The ipm file");
DEFINE_string(output_video, "",
    "The output video file");
DEFINE_double(confidence_threshold, 0.2,
    "Only store detections with score higher than the threshold.");
DEFINE_int32(width, 0, "Set image width");
DEFINE_int32(height, 0, "Set image height");
DEFINE_int32(fps, 0, "Set capture fps");
DEFINE_int32(DEBUG, 0, "Whether need to debug");
DEFINE_int32(SERIAL_OPEN, 1, "Whether need to open serial port");
DEFINE_int32(VIDEO_STORE, 0, "Whether need to store video");

//-------------------------用于存储通过gflag传入的值-------------------------------
string output_video, IPM_file;
int camera_id, im_width, im_height, im_fps, _DEBUG, _SERIAL_OPEN, _VIDEO_STORE;

//-------------------------中间信息打印至文件,用于调试-------------------------------
ofstream location_out;
//location_out.open("location_out.txt",std::ios::out | std::ios::app);
ofstream location_out_new;
//location_out_new.open("location_out_new.txt");
ofstream save_distance("tesult.txt");
ofstream buffer("buffer.txt");

double _ANGLEl;		//add by nh
double _ANGLEr;		//add by nh



//--------------------------------目标检测类别-----------------------------------

string CLASSES[] = {"__background__", "bicycle", "bus", "car", "truck",
					"van", "motorbike", "person", "tricycle"};

//-------------------------------目标BBox颜色表------------------------------------------
cv::Scalar COLOR[13]={cv::Scalar(101,67,254),cv::Scalar(18,87,220),
					  cv::Scalar(154,157,252),cv::Scalar(185,235,35),cv::Scalar(151,191,29),
					  cv::Scalar(76,29,78),cv::Scalar(202,188,62),cv::Scalar(171,90,0),
					  cv::Scalar(205,55,14),cv::Scalar(53,194,107),cv::Scalar(225,169,26),
					  cv::Scalar(31,31,197),cv::Scalar(0,0,255)};

//global variable for sharing between threads
bool DETECT_FLAG = true;
bool OVER = false;//视频流结束标志
vector<cv::Point> lanes; //理想车道线识别结果（四个坐标）
back_lines final_lanes;//车道线识别返回值

cv::Mat frame;
cv::Mat road;
cv::Mat img;
cv::Mat frame_global;
cv::Mat img_temp;
cv::Mat img_temp2;
cv::Mat canvas;
cv::Mat canvas_detect;

bool NEW_DETECT = false;
bool TRACK_FLAG = false;
bool NEW_IMAGE = false;

//Crop image region
Rect crop_region = Rect(Point(0,0), Point(640,480));

//---------------------------SSD初始化---------------------------


string model_prototxt_file = "/home/gchy/shikewei-all3.8/models/VGGNet/ADAS/SSD_300x300_mixture/deploy.prototxt";
string trained_caffemodel_file = "/home/gchy/shikewei-all3.8/models/VGGNet/ADAS/SSD_300x300_mixture/VGG_ADAS_SSD_300x300_iter_60000.caffemodel";


SSD_Detector detector(model_prototxt_file, trained_caffemodel_file);

vector<SSD_Detector::Object> Obj_pool_temp;
vector<SSD_Detector::Object> track_pool_temp;
vector<Tracking> tracking_obj_pool; //for tracking
vector<SSD_Detector::Object> tracked_obj_pool;


typedef struct Light_object
{
	int label;
	Point center_pos = Point(0,0);
}; 

Light_object TargetRes[8];
int TargetCnt=0;
//-------------------------CAN报文中目标检测信息----------------------------------
static char can_buffer10[10]={0x00,0x20,0x00};
static char can_buffer11[10]={0x00,0x21,0x00};
static char can_buffer12[10]={0x00,0x22,0x00};
static char can_buffer13[10]={0x00,0x23,0x00};
static char can_buffer14[10]={0x00,0x24,0x00};
static char can_buffer15[10]={0x00,0x25,0x00};
static char can_buffer16[10]={0x00,0x26,0x00};
static char can_buffer17[10]={0x00,0x27,0x00};
char * pTargetCanBuffer[8] =
{can_buffer10,can_buffer11,can_buffer12,can_buffer13,can_buffer14,can_buffer15,can_buffer16,can_buffer17};

//=====================================目标检测框绘制===========================================
void draw_bb_top(cv::Mat &img_, string &name, cv::Point &pt_lt,
                                  cv::Point &pt_br, cv::Scalar &color, cv::Point2d &dist_, string &pos)
{
	cv::rectangle(img_,cv::Point(pt_lt.x,pt_lt.y-15),pt_br,color,2);
	cv::rectangle(img_,cv::Point(pt_lt.x,pt_lt.y-15),
                    cv::Point(pt_lt.x + name.length()*15,pt_lt.y),color,-1);
	cv::putText(img_,name,cv::Point(pt_lt.x+7,pt_lt.y-4), cv::FONT_HERSHEY_SIMPLEX,
							0.6,cv::Scalar(255,255,255),1,8);//lt means lefttop
	
	
}


void CIPV_Judge_for_drawing(vector<SSD_Detector::Object>& obj_pool,cv::Mat& frame_)
{

    for(size_t i=0;i<obj_pool.size();i++)
    {
        cv::Point lt = obj_pool[i].boundingbox.tl();//检测的bbox左上角坐标
        cv::Point rb = Point(lt.x+obj_pool[i].boundingbox.width, lt.y+obj_pool[i].boundingbox.height);//bbox右下角坐标
            //-------------------------------------------非危险目标------------------------------------------------------
            draw_bb_top(frame_, CLASSES[obj_pool[i].label], lt, rb, COLOR[obj_pool[i].label], obj_pool[i].dist, obj_pool[i].position);
    }
}




//=============================CAN报文发送线程=====================================
void can_transfer(int fd2)
{
  int nread2;
  while(1)
  {
    usleep(50000);
	nread2 = write(fd2, can_buffer10, 10);
	usleep(1500);
	nread2 = write(fd2, can_buffer11, 10);
	usleep(1500);
	nread2 = write(fd2, can_buffer12, 10);
	usleep(1500);
	nread2 = write(fd2, can_buffer13, 10);
	usleep(1500);
	nread2 = write(fd2, can_buffer14, 10);
	usleep(1500);
	nread2 = write(fd2, can_buffer15, 10);
	usleep(1500);
	nread2 = write(fd2, can_buffer16, 10);
	usleep(1500);
	nread2 = write(fd2, can_buffer17, 10);
	usleep(1500);
  }
}


void func_video_input()
{

    //-------------------------从摄像头/视频中读入图像流-------------------------
  	VideoCapture inputVideo(0);

  	while(true)
	{

		inputVideo >> frame;

        usleep(5000);//加入短暂延时,防止循环过快导致卡顿
		//LaneRes.IsCrossingLane = 0;
    	if(!frame.empty())
		{
			if(framecount>1000000) framecount=10;

     		double t = (double)cv::getTickCount();


			if(DETECT_FLAG == true & framecount > 1)
			{

				DETECT_FLAG = false;

				img_temp = frame.clone();


			}
            framecount++;
		}
    else OVER = true;//Video end
	}

}

void SSD_detect()
{
    Caffe::SetDevice(0);
    Caffe::set_mode(Caffe::GPU);
    //VideoCapture inputVideo("/home/chaowei/Videos/0918/2/output_original.avi");
	while(1)
	{

		if(!frame.empty())
		{
			double t = (double) cv::getTickCount();

			detector.detect(frame, 300, FLAGS_confidence_threshold);

            t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();

                cout << "---------Obj_pool.size>" << detector.Obj_pool.size()<< "<---------"<< endl;
                if(DETECT_FLAG == true & framecount > 1)
			   {

				DETECT_FLAG = false;
				img_temp = frame.clone();
			   }
               framecount++;

		}
	}
}

int main(int argc, char** argv)
{
  	::google::InitGoogleLogging(argv[0]);
  	FLAGS_alsologtostderr = 1;
  	gflags::SetUsageMessage("Read video file and save video.\n"
        "Usage:\n"
        "    TAGE_Camera --model_file *.prototxt --weights_file *.caffemodel\n"
        "         --camera_id 0 or 1 --output_video *.avi --ipm_file *.xml\n"
        "         --confidence_threshold 0.2\n");
 	 gflags::ParseCommandLineFlags(&argc, &argv, true);

    //-------------------gflag传入的值---------------------
  	camera_id = FLAGS_camera_id;
  	output_video = FLAGS_output_video;
  	IPM_file = FLAGS_ipm_file;
  	im_width = FLAGS_width;
  	im_height = FLAGS_height;
  	im_fps = FLAGS_fps;
  	_DEBUG = FLAGS_DEBUG;
  	_SERIAL_OPEN = FLAGS_SERIAL_OPEN;
  	_VIDEO_STORE = FLAGS_VIDEO_STORE;
    //caffe设置
	Caffe::SetDevice(0);
	Caffe::set_mode(Caffe::GPU);
  	boost::thread thrd(&func_video_input);
	//----------------------打开串口------------------
	int fd2,nread2,i2;
	if(_SERIAL_OPEN)
	{
        //串口开启
		if((fd2=open_port(fd2,1))<0)
		{
		    perror("Open USB Serial Port Error !");
		}
        //串口设置
		if((i2=set_opt(fd2,115200,8,'N',1))<0)
		{
		    perror("USB Serial Port set_opt Error");
		}
        //若正常打开串口,则开启CAN报文传输线程
		boost::thread thrd2(&can_transfer,fd2);
  	}

	boost::thread thrd3(&SSD_detect);

	if(_VIDEO_STORE)
	{
		time_t timep;
    struct tm* p;
    char* str;
    char dir_path[200];
    char original_file[200];
    timep=time(NULL);
    p=localtime(&timep);	    
    str=asctime(p); 
    for(int i=0;str[i]!='\0';i++)
    {
			if(str[i]==' ')str[i]='_';
			if(str[i]==':')str[i]='_';
			if(str[i]=='\n')str[i]='\0';
    }
    sprintf(dir_path,"/home/cao/MyFile/codes/shikewei-all3.8/%s",str);
    mkdir(dir_path,00700);
    sprintf(original_file,"%s/original.avi",dir_path);
    originalVideo.open(original_file, CV_FOURCC('M','J','P','G'),20, Size(640,480));
	}
while(true)
  	{


		
    	if(!img_temp.empty())
    	{
    	    double t = (double) cv::getTickCount();
            //目标检测与跟踪
            Obj_pool_temp = detector.Obj_pool;
            Object_track();
            CIPV_Judge_for_drawing(detector.Obj_pool,img_temp);

			DETECT_FLAG = true;
			img_temp.~Mat();
            t = ((double) cv::getTickCount() - t) / cv::getTickFrequency();
		}
		if(OVER==true)
		{
			break;
      //close(fd2);
		}
	}
	return 0;
}
