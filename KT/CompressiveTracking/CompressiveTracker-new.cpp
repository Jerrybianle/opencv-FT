#include "CompressiveTracker.h"
#include <math.h>
#include <iostream>
using namespace cv;
using namespace std;

//------------------------------------------------
CompressiveTracker::CompressiveTracker(void)
{	
	featureNum = 64;	// number of all weaker classifiers, i.e,feature pool
	rOuterPositive = 4;	// radical scope of positive samples
	rSearchWindow = 25; // size of search window
	muPositive = vector<float>(featureNum, 0.0f);
	muNegative = vector<float>(featureNum, 0.0f);
	sigmaPositive = vector<float>(featureNum, 1.0f);
	sigmaNegative = vector<float>(featureNum, 1.0f);
	learnRate = 0.85f;	// Learning rate parameter
//-----------------------------------------------------
	value=0;
	numCentroids=4;
	for(int i=0;i<numCentroids;i++)
     {
		 labels.push_back(i);
     }

}

CompressiveTracker::~CompressiveTracker(void)
{
}

//get kmodle
void CompressiveTracker::KFeature(Mat& _frame,Rect& _objectBox,vector<Rect>& _sampleBox)
{
   Mat rngimg,centroids,dstr,dstb;
   Mat res(1,1,CV_32F);
   float da;
   
	
//去相关性，得到转换后的RIMG白化问题
	Mat rimg=_frame( _objectBox);
    resize(rimg,rimg,Size(32,32));
	meanvalue=mean(rimg);
    rimg.convertTo(rimg,CV_32FC1,1.0,-meanvalue[0]);
	
	

//提取图片块考虑随机提取，大小问题
	
	for(int i=0;i<rimg.rows/2;i++)
		for(int j=0;j<rimg.cols/2;j++)
		{
			Rect rngroi(i*2,j*2,2,2);
			Mat rngtemproi=rimg(rngroi);
			Mat rngtemp=rngtemproi.clone();
			Mat rngtempres=rngtemp.reshape(0,1);	
			rngimg.push_back(rngtempres);
			
		}
//进行聚类学习次数太少考虑其他的方法比如EM或者矢量量化
   kmeans(rngimg,numCentroids,labels,TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),1,KMEANS_USE_INITIAL_LABELS,centroids);
//与目标图片进行卷积得到特征模型考虑FFT加快速度
  
   if(Kmodle.empty())
    {
	   Kmodle=centroids;
    }
   else
    {
		matchTemplate( Kmodle,centroids, res, CV_TM_CCORR_NORMED);//考虑范数
		if( ((float *)(res.data))[0]<value ) //自动获取阈值
		{
             filter2D(rimg,dstr,-1,centroids,Point(-1,-1));
	         Mat bimg=_frame( _sampleBox[_sampleBox.size()-1] );
		     resize(bimg,bimg,Size(32,32));
			 meanvalue=mean(bimg);
             bimg.convertTo(bimg,CV_32FC1,1.0,-meanvalue[0]);
             filter2D(bimg,dstb,-1,centroids,Point(-1,-1));
             //double da=norm(dstr,dstb,NORM_L2);
	         matchTemplate(dstr,dstb,res,CV_TM_CCORR_NORMED);
	         da=((float *)(res.data))[0];
             addWeighted(Kmodle,da,centroids,1-da,0.0,Kmodle);

        }
		else
		{
			value=((float *)(res.data))[0];
		}

    }

	
}


void CompressiveTracker::sampleRect(Mat& _image, Rect& _objectBox, float _rInner, float _rOuter, int _maxSampleNum, vector<Rect>& _sampleBox)
/* Description: compute the coordinate of positive and negative sample image templates
   Arguments:
   -_image:        processing frame
   -_objectBox:    recent object position 
   -_rInner:       inner sampling radius
   -_rOuter:       Outer sampling radius
   -_maxSampleNum: maximal number of sampled images
   -_sampleBox:    Storing the rectangle coordinates of the sampled images.
*/
{
	int rowsz = _image.rows - _objectBox.height - 1;
	int colsz = _image.cols - _objectBox.width - 1;
	float inradsq = _rInner*_rInner;
	float outradsq = _rOuter*_rOuter;

  	
	int dist;

	int minrow = max(0,(int)_objectBox.y-(int)_rInner);
	int maxrow = min((int)rowsz-1,(int)_objectBox.y+(int)_rInner);
	int mincol = max(0,(int)_objectBox.x-(int)_rInner);
	int maxcol = min((int)colsz-1,(int)_objectBox.x+(int)_rInner);
    
	
	
	int i = 0;

	float prob = ((float)(_maxSampleNum))/(maxrow-minrow+1)/(maxcol-mincol+1);

	int r;
	int c;
    
    _sampleBox.clear();//important
    Rect rec(0,0,0,0);

	for( r=minrow; r<=(int)maxrow; r++ )
		for( c=mincol; c<=(int)maxcol; c++ ){
			dist = (_objectBox.y-r)*(_objectBox.y-r) + (_objectBox.x-c)*(_objectBox.x-c);

			if( rng.uniform(0.,1.)<prob && dist < inradsq && dist >= outradsq ){

                rec.x = c;
				rec.y = r;
				rec.width = _objectBox.width;
				rec.height= _objectBox.height;
				
                _sampleBox.push_back(rec);				
				
				i++;
			}
		}
	
		_sampleBox.resize(i);
		
}

void CompressiveTracker::sampleRect(Mat& _image, Rect& _objectBox, float _srw, vector<Rect>& _sampleBox)
/* Description: Compute the coordinate of samples when detecting the object.*/
{
	int rowsz = _image.rows - _objectBox.height - 1;
	int colsz = _image.cols - _objectBox.width - 1;
	float inradsq = _srw*_srw;	
	

	int dist;

	int minrow = max(0,(int)_objectBox.y-(int)_srw);
	int maxrow = min((int)rowsz-1,(int)_objectBox.y+(int)_srw);
	int mincol = max(0,(int)_objectBox.x-(int)_srw);
	int maxcol = min((int)colsz-1,(int)_objectBox.x+(int)_srw);

	int i = 0;

	int r;
	int c;

	Rect rec(0,0,0,0);
    _sampleBox.clear();//important

	for( r=minrow; r<=(int)maxrow; r++ )
		for( c=mincol; c<=(int)maxcol; c++ ){
			dist = (_objectBox.y-r)*(_objectBox.y-r) + (_objectBox.x-c)*(_objectBox.x-c);

			if( dist < inradsq ){

				rec.x = c;
				rec.y = r;
				rec.width = _objectBox.width;
				rec.height= _objectBox.height;

				_sampleBox.push_back(rec);				

				i++;
			}
		}
	
		_sampleBox.resize(i);

}
// Compute the features of samples
void CompressiveTracker::getFeatureValue(Mat& _frame, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue)
{
	//池化问题
	Mat collmat,dst,dstresp,sampleFeatureValue;
	for(int i=0;i<_sampleBox.size();i++)
	  {
		   Mat temp=_frame(_sampleBox[i]);
           resize(temp,temp,Size(32,32));
		   meanvalue=mean(temp);
           temp.convertTo(temp,CV_32FC1,1.0,-meanvalue[0]);
		   collmat.push_back(temp);
	  }
	filter2D(collmat,dst,-1,Kmodle,Point(-1,-1));
    for(int i=0;i<_sampleBox.size();i++)
	  {
		   Mat dsttemp=dst.rowRange(i*32,(i+1)*32).clone();
		   resize(dsttemp,dsttemp,Size(8,8));
		   dstresp=dsttemp.reshape(0,1);
		   sampleFeatureValue.push_back(dstresp);
	  }
	_sampleFeatureValue=sampleFeatureValue.t();
	
}


// Update the mean and variance of the gaussian classifier
void CompressiveTracker::classifierUpdate(Mat& _sampleFeatureValue, vector<float>& _mu, vector<float>& _sigma, float _learnRate)
{
	Scalar muTemp;
	Scalar sigmaTemp;
    
	for (int i=0; i<featureNum; i++)
	{
		meanStdDev(_sampleFeatureValue.row(i), muTemp, sigmaTemp);
	   
		_sigma[i] = (float)sqrt( _learnRate*_sigma[i]*_sigma[i]	+ (1.0f-_learnRate)*sigmaTemp.val[0]*sigmaTemp.val[0] 
		+ _learnRate*(1.0f-_learnRate)*(_mu[i]-muTemp.val[0])*(_mu[i]-muTemp.val[0]));	// equation 6 in paper

		_mu[i] = _mu[i]*_learnRate + (1.0f-_learnRate)*muTemp.val[0];	// equation 6 in paper
	}
}

// Compute the ratio classifier 
void CompressiveTracker::radioClassifier(vector<float>& _muPos, vector<float>& _sigmaPos, vector<float>& _muNeg, vector<float>& _sigmaNeg,
										 Mat& _sampleFeatureValue, float& _radioMax, int& _radioMaxIndex)
{
	float sumRadio;
	_radioMax = -FLT_MAX;
	_radioMaxIndex = 0;
	float pPos;
	float pNeg;
	int sampleBoxNum = _sampleFeatureValue.cols;

	for (int j=0; j<sampleBoxNum; j++)
	{
		sumRadio = 0.0f;
		for (int i=0; i<featureNum; i++)
		{
			pPos = exp( (_sampleFeatureValue.at<float>(i,j)-_muPos[i])*(_sampleFeatureValue.at<float>(i,j)-_muPos[i]) / -(2.0f*_sigmaPos[i]*_sigmaPos[i]+1e-30) ) / (_sigmaPos[i]+1e-30);
			pNeg = exp( (_sampleFeatureValue.at<float>(i,j)-_muNeg[i])*(_sampleFeatureValue.at<float>(i,j)-_muNeg[i]) / -(2.0f*_sigmaNeg[i]*_sigmaNeg[i]+1e-30) ) / (_sigmaNeg[i]+1e-30);
			sumRadio += log(pPos+1e-30) - log(pNeg+1e-30);	// equation 4
		}
		if (_radioMax < sumRadio)
		{
			_radioMax = sumRadio;
			_radioMaxIndex = j;
		}
	}
}
void CompressiveTracker::init(Mat& _frame, Rect& _objectBox)
{
	// compute feature template 100
	sampleRect(_frame, _objectBox, rSearchWindow*1.5, rOuterPositive+4.0, 100, sampleNegativeBox);
	KFeature(_frame,_objectBox,sampleNegativeBox);

	// compute sample templates 1000000
	sampleRect(_frame, _objectBox, rOuterPositive,0,1000000,samplePositiveBox);

	getFeatureValue(_frame, samplePositiveBox, samplePositiveFeatureValue);
	getFeatureValue(_frame, sampleNegativeBox, sampleNegativeFeatureValue);
	classifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	classifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);
}
void CompressiveTracker::processFrame(Mat& _frame, Rect& _objectBox)
{
	// predict
	sampleRect(_frame, _objectBox, rSearchWindow,detectBox);
	getFeatureValue(_frame, detectBox, detectFeatureValue);
	int radioMaxIndex;
	float radioMax;
	radioClassifier(muPositive, sigmaPositive, muNegative, sigmaNegative, detectFeatureValue, radioMax, radioMaxIndex);
	_objectBox = detectBox[radioMaxIndex];

	// update
    init(_frame, _objectBox);
}