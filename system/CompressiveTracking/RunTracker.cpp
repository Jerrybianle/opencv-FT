#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include "Tracker.h"
#include"Detecter.h"

using namespace cv;
using namespace std;

Rect box; // tracking object
bool drawing_box = false;
bool gotBB = false;	// got tracking box or not
bool fromfile = false;
string video;

void readBB(char* file)	// get tracking box from file
{
	ifstream tb_file (file);    //read file
	string line;
	getline(tb_file, line);      //按行读入文件
	istringstream linestream(line);     //串流读入
	string x1, y1, w1, h1;
	getline(linestream, x1, ',');    //从linestream输入流中读取字符存入x1,遇到“，”终止
	getline(linestream, y1, ',');
	getline(linestream, w1, ',');
	getline(linestream, h1, ',');
	int x = atoi(x1.c_str());       //将字符串转成整数
	int y = atoi(y1.c_str());
	int w = atoi(w1.c_str());
	int h = atoi(h1.c_str());
	box = Rect(x, y, w, h);
}

// tracking box mouse callback
//手动画出目标框
void mouseHandler(int event, int x, int y, int flags, void *param)
{
	switch (event)
	{
	case CV_EVENT_MOUSEMOVE:
		if (drawing_box)
		{
			box.width = x - box.x;
			box.height = y - box.y;
		}
		break;
	case CV_EVENT_LBUTTONDOWN:
		drawing_box = true;
		box = Rect(x, y, 0, 0);
		break;
	case CV_EVENT_LBUTTONUP:
		drawing_box = false;
		if (box.width < 0)
		{
			box.x += box.width;
			box.width *= -1;
		}
		if( box.height < 0 )
		{
			box.y += box.height;
			box.height *= -1;
		}
		gotBB = true;
		break;
	default:
		break;
	}
}

void print_help(void)
{
	printf("use:\n     welcome to use CompressiveTracking\n");
	printf("Kaihua Zhang's paper:Real-Time Compressive Tracking\n");
	printf("C++ implemented by yang xian\nVersion: 1.0\nEmail: yang_xian521@163.com\nDate:	2012/08/03\n\n");
	printf("-v    source video\n-b        tracking box file\n");
}

void read_options(int argc, char** argv, VideoCapture& capture)
{
	for (int i=0; i<argc; i++)
	{
		if (strcmp(argv[i], "-b") == 0)	// read tracking box from file
		{
			if (argc>i)
			{
				readBB(argv[i+1]);
				gotBB = true;
			}
			else
			{
				print_help();
			}
		}
		if (strcmp(argv[i], "-v") == 0)	// read video from file
		{
			if (argc > i)
			{
				video = string(argv[i+1]);
				capture.open(video);
				fromfile = true;
			}
			else
			{
				print_help();
			}
		}
	}
}

int main(int argc, char * argv[])
{
	cout<< "Author: Li Pengfei"<<endl;
	cout<< "Version: 1.6"<<endl;
	cout<< "Date: 2016/03/26"<<endl;
	VideoCapture capture;
	capture.open(0);
	// Read options
	read_options(argc, argv, capture);
	// Init camera
	if (!capture.isOpened())
	{
		cout << "capture device failed to open!" << endl;
		return 1;
	}
	// Register mouse callback to draw the tracking box
	namedWindow("FT", CV_WINDOW_AUTOSIZE);
	setMouseCallback("FT", mouseHandler, NULL);
	// CT framework
	Tracker tracker;
	Detecter t;

	Mat frame;
	Mat last_gray;
	Mat first;
	if (fromfile)
	{
		capture >> frame;
		cvtColor(frame, last_gray, CV_RGB2GRAY);
		frame.copyTo(first);
	}
	else
	{
		capture.set(CV_CAP_PROP_FRAME_WIDTH, 340);
		capture.set(CV_CAP_PROP_FRAME_HEIGHT, 240);
	}

	// Initialization
	while(!gotBB)
	{
		if (!fromfile)
		{
			capture >> frame;
		}
		else
		{
			first.copyTo(frame);
		}
		cvtColor(frame, last_gray, CV_RGB2GRAY);
		rectangle(frame, box, Scalar(0,0,255));
		imshow("FT", frame);
		if (cvWaitKey(33) == 27) {	return 0; }
	}
	
	// Remove callback
	setMouseCallback("FT", NULL, NULL);
	printf("Initial Tracking Box = x:%d y:%d h:%d w:%d\n", box.x, box.y, box.width, box.height);
	// CT initialization
	
	tracker.init(last_gray, box);
	t.init(last_gray,box);
	
	// Run-time
	Mat current_gray;
	bool trackstu=true;

	while(capture.read(frame))
	{
		// get frame
		cvtColor(frame, current_gray, CV_RGB2GRAY);
		// Process Frame
		if(trackstu)
		{
		   tracker.processFrame(current_gray, box,trackstu);
		   // Draw Points
		   rectangle(frame, box, Scalar(0,0,255));
		}
		else
		{
			t.detectimg(current_gray,box,trackstu);
			if(trackstu)
				tracker.detectLearn(current_gray,box);

		}
		// Display
		imshow("FT", frame);
		//printf("Current Tracking Box = x:%d y:%d h:%d w:%d\n", box.x, box.y, box.width, box.height);

		if (cvWaitKey(33) == 27) {	break; }
	}
	return 0;
}