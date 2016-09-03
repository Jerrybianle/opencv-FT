/************************************************************************
* File:	RunTracker.cpp
* Brief: C++ demo for Kaihua Zhang's paper:"Real-Time Compressive Tracking"
* Version: 1.0
* Author: Yang Xian
* Email: yang_xian521@163.com
* Date:	2012/08/03
* History:
************************************************************************/
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdio>
#include "KT.h"


using namespace cv;
using namespace std;

Rect box; // tracking object
bool drawing_box = false;
bool gotBB = false;	// got tracking box or not
bool fromfile = true;
string video;
int ki=0;
char name[256];

void readBB(char* file)	// get tracking box from file
{
	ifstream tb_file (file);
	string line;
	getline(tb_file, line);
	istringstream linestream(line);
	string x1, y1, w1, h1;
	getline(linestream, x1, ',');
	getline(linestream, y1, ',');
	getline(linestream, w1, ',');
	getline(linestream, h1, ',');
	int x = atoi(x1.c_str());
	int y = atoi(y1.c_str());
	int w = atoi(w1.c_str());
	int h = atoi(h1.c_str());
	box = Rect(x, y, w, h);
}

// tracking box mouse callback
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
		box.x=142;
		box.y=125;
		box.width=90;
		box.height=39;
		
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
	VideoCapture capture;
	capture.open("car.avi");
	// Read options
	read_options(argc, argv, capture);
	// Init camera
	if (!capture.isOpened())
	{
		cout << "capture device failed to open!" << endl;
		return 1;
	}
	// Register mouse callback to draw the tracking box
	namedWindow("CT", CV_WINDOW_AUTOSIZE);
	setMouseCallback("CT", mouseHandler, NULL);
	// CT framework
	KT ct;

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
		imshow("CT", frame);
		if (cvWaitKey(33) == 27) {	return 0; }
	}
	
	// Remove callback
	setMouseCallback("CT", NULL, NULL);
	printf("Initial Tracking Box = x:%d y:%d h:%d w:%d\n", box.x, box.y, box.width, box.height);
	// CT initialization
	ct.init(last_gray, box);

	// Run-time
	Mat current_gray;

	while(capture.read(frame))
	{
		// get frame
		cvtColor(frame, current_gray, CV_RGB2GRAY);
		// Process Frame
		ct.processFrame(current_gray, box);
		// Draw Points
		rectangle(frame, box, Scalar(0,0,255),6);
		// Display
		imshow("CT", frame);

		sprintf(name,"D://c1//%d.jpg",ki);
		//printf("Current Tracking Box = x:%d y:%d h:%d w:%d\n", box.x, box.y, box.width, box.height);
		imwrite(name,frame);
		ki++;

		if (cvWaitKey(33) == 27) {	break; }
	}
	return 0;
}