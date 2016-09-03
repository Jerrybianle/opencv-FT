#include "Tracker.h"
#include <math.h>
#include <iostream>
using namespace cv;
using namespace std;

//------------------------------------------------
Tracker::Tracker(void)
{
	featureMinNumRect = 2;
	featureMaxNumRect = 4;	// number of rectangle from 2 to 4
	featureNum = 50;	// number of all weaker classifiers, i.e,feature pool
	rOuterPositive = 4;	// radical scope of positive samples
	rSearchWindow = 25; // size of search window
	muPositive = vector<float>(featureNum, 0.0f);
	muNegative = vector<float>(featureNum, 0.0f);
	sigmaPositive = vector<float>(featureNum, 1.0f);
	sigmaNegative = vector<float>(featureNum, 1.0f);
	learnRate = 0.85f;	// Learning rate parameter
	ThrC=0;
}

Tracker::~Tracker(void)
{
}


void Tracker::HaarFeature(Rect& _objectBox, int _numFeature)
/*Description: compute Haar features
  Arguments:
  -_objectBox: [x y width height] object rectangle
  -_numFeature: total number of features.The default is 50.
*/
{
	features = vector<vector<Rect>>(_numFeature, vector<Rect>());
	featuresWeight = vector<vector<float>>(_numFeature, vector<float>());
	
	int numRect;
	Rect rectTemp;
	float weightTemp;
      
	for (int i=0; i<_numFeature; i++)
	{
		numRect = cvFloor(rng.uniform((double)featureMinNumRect, (double)featureMaxNumRect));    //cvFloor���ز����ڲ������������ֵ
		                                                                                         //rng.uniform����һ��featureMinNumRect��featureMaxNumRect���double�����������2��3
	    
		//int c = 1;
		for (int j=0; j<numRect; j++)    //��Ŀ�����ȡ��������
		{
			
			rectTemp.x = cvFloor(rng.uniform(0.0, (double)(_objectBox.width - 3)));
			rectTemp.y = cvFloor(rng.uniform(0.0, (double)(_objectBox.height - 3)));
			rectTemp.width = cvCeil(rng.uniform(0.0, (double)(_objectBox.width - rectTemp.x - 2)));
			rectTemp.height = cvCeil(rng.uniform(0.0, (double)(_objectBox.height - rectTemp.y - 2)));
			features[i].push_back(rectTemp);

			weightTemp = (float)pow(-1.0, cvFloor(rng.uniform(0.0, 2.0))) / sqrt(float(numRect));
            //weightTemp = (float)pow(-1.0, c);    //c = 0 or 1
			
			featuresWeight[i].push_back(weightTemp);
           
		}
	}
}


void Tracker::sampleRect(Mat& _image, Rect& _objectBox, float _rInner, float _rOuter, int _maxSampleNum, vector<Rect>& _sampleBox)
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

void Tracker::sampleRect(Mat& _image, Rect& _objectBox, float _srw, vector<Rect>& _sampleBox)
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
void Tracker::getFeatureValue(Mat& _imageIntegral, vector<Rect>& _sampleBox, Mat& _sampleFeatureValue)
{
	int sampleBoxSize = _sampleBox.size();
	_sampleFeatureValue.create(featureNum, sampleBoxSize, CV_32F);
	float tempValue;
	int xMin;
	int xMax;
	int yMin;
	int yMax;

	for (int i=0; i<featureNum; i++)
	{
		for (int j=0; j<sampleBoxSize; j++)
		{
			tempValue = 0.0f;
			for (size_t k=0; k<features[i].size(); k++)
			{
				xMin = _sampleBox[j].x + features[i][k].x;
				xMax = _sampleBox[j].x + features[i][k].x + features[i][k].width;
				yMin = _sampleBox[j].y + features[i][k].y;
				yMax = _sampleBox[j].y + features[i][k].y + features[i][k].height;
				tempValue += featuresWeight[i][k] * 
					(_imageIntegral.at<float>(yMin, xMin) +
					_imageIntegral.at<float>(yMax, xMax) -
					_imageIntegral.at<float>(yMin, xMax) -
					_imageIntegral.at<float>(yMax, xMin));
			}
			_sampleFeatureValue.at<float>(i,j) = tempValue;
		}
	}
}

// Update the mean and variance of the gaussian classifier
void Tracker::classifierUpdate(Mat& _sampleFeatureValue, vector<float>& _mu, vector<float>& _sigma, float _learnRate)
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
void Tracker::radioClassifier(vector<float>& _muPos, vector<float>& _sigmaPos, vector<float>& _muNeg, vector<float>& _sigmaNeg,
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
//--------------------------------------------------------------
void Tracker::getPattern(Mat&_img, Mat&_pattern)
{
   Scalar mean ,stdev;
  resize(_img,_pattern,Size(15,15));
  meanStdDev(_pattern,mean,stdev);  
  _pattern.convertTo(_pattern,CV_32F);
  _pattern = _pattern-mean.val[0];
}
//---------------------------------------------------------------
void Tracker::NNConf( Mat& example,float& rsconf)
{
  if (PSample.empty())
  { 
      rsconf = 0; 
      return;
  }
  if (NSample.empty())
  { 
      rsconf = 1;   
      return;
  }
  Mat ncc(1,1,CV_32F);
  float nccP,maxP=0;
  float nccN, maxN=0;

  for (int i=0;i<PSample.size();i++)
  {
      matchTemplate(PSample[i],example,ncc,CV_TM_CCORR_NORMED);      
      nccP=(((float*)ncc.data)[0]+1)*0.5; 
      if(nccP > maxP)
          maxP=nccP; 
  }
  
  for (int i=0;i<NSample.size();i++)
  {
      matchTemplate(NSample[i],example,ncc,CV_TM_CCORR_NORMED); 
      nccN=(((float*)ncc.data)[0]+1)*0.5;
      if(nccN > maxN)
        maxN=nccN;
  }

  //Measure Relative Similarity

  float dN=1-maxN;
  float dP=1-maxP;
  rsconf = (float)dN/(dN+dP);
}

//-------------------------ѧϰģ��------------------------------------------------
void Tracker::Learn(Mat& _frame,Rect& _objectBox)
{
	integral(_frame,imageIntegral, CV_32F);
	// update
	sampleRect(_frame, _objectBox, rOuterPositive, 0.0, 1000000, samplePositiveBox);
	sampleRect(_frame, _objectBox, rSearchWindow*1.5, rOuterPositive+4.0, 100, sampleNegativeBox);
	
	getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);
	getFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue);
	classifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	classifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);
   //---------------------------------------------------------------
	Mat Pox,Nax;
	vector<Mat>NSampleT;
	if(PSample.size()==20)
	{
       random_shuffle(PSample.begin(),PSample.end());
	   PSample.resize(PSample.size()/2);
	}
	getPattern(_frame(_objectBox),Pox);
	PSample.push_back(Pox);

	NSample.clear();
	for(int i=0;i<sampleNegativeBox.size();i++)
	{
	  if( bbOverlap(_objectBox,sampleNegativeBox[i])<0.5)
	  {
         getPattern(_frame(sampleNegativeBox[i]),Nax);
	     NSample.push_back(Nax);
	  }
	}
	int half = (int)NSample.size()*0.5f;
    NSampleT.assign(NSample.begin()+half,NSample.end());
    NSample.resize(half);
	//------------------------------------------------------
    float conf;
	ThrC=0;
    for (int i=0;i<NSampleT.size();i++)
	{
      NNConf(NSampleT[i],conf);
      if (conf>ThrC)
        ThrC=conf; 
	}
	NSampleT.clear();

}
//-----------------------------------------------------------------------------------------------

//-------------------------���ѧϰģ��------------------------------------------------
void Tracker::detectLearn(Mat& _frame,Rect& _objectBox)
{
	integral(_frame,imageIntegral, CV_32F);
	// update
	sampleRect(_frame, _objectBox, rOuterPositive, 0.0, 1000000, samplePositiveBox);
	sampleRect(_frame, _objectBox, rSearchWindow*1.5, rOuterPositive+4.0, 100, sampleNegativeBox);
	
	getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);
	getFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue);
	classifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	classifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);
   //---------------------------------------------------------------
	Mat Pox,Nax;
	vector<Mat>NSampleT;
    PSample.clear();
	getPattern(_frame(_objectBox),Pox);
	PSample.push_back(Pox);

	NSample.clear();
	for(int i=0;i<sampleNegativeBox.size();i++)
	{
	  if( bbOverlap(_objectBox,sampleNegativeBox[i])<0.5)
	  {
         getPattern(_frame(sampleNegativeBox[i]),Nax);
	     NSample.push_back(Nax);
	  }
	}
	int half = (int)NSample.size()*0.5f;
    NSampleT.assign(NSample.begin()+half,NSample.end());
    NSample.resize(half);
	//------------------------------------------------------
    float conf;
	ThrC=0;
    for (int i=0;i<NSampleT.size();i++)
	{
      NNConf(NSampleT[i],conf);
      if (conf>ThrC)
        ThrC=conf; 
	}
	NSampleT.clear();
}
//-----------------------------------------------------------------------------------------------



//---------------------------------------------------------------
void Tracker::init(Mat& _frame, Rect& _objectBox)
{
	
	// compute feature template
	HaarFeature(_objectBox, featureNum);

	// compute sample templates
	sampleRect(_frame, _objectBox, rOuterPositive, 0, 1000000, samplePositiveBox);
	sampleRect(_frame, _objectBox, rSearchWindow*1.5, rOuterPositive+4.0, 100, sampleNegativeBox);

	integral(_frame, imageIntegral, CV_32F);

	getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);
	getFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue);
	classifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	classifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);
	 //-------------------------------------------------------------------------------------
	Mat Pox,Nax;
	vector<Mat>NSampleT;
	getPattern(_frame(_objectBox),Pox);
	PSample.push_back(Pox);
	for(int i=0;i<sampleNegativeBox.size();i++)
	{
	  if( bbOverlap(_objectBox,sampleNegativeBox[i])<0.5)
	  {
         getPattern(_frame(sampleNegativeBox[i]),Nax);
	     NSample.push_back(Nax);
	  }
	}
	int half = (int)NSample.size()*0.5f;
    NSampleT.assign(NSample.begin()+half,NSample.end());
    NSample.resize(half);
	//------------------------------------------------------
    float conf;
    for (int i=0;i<NSampleT.size();i++)
	{
      NNConf(NSampleT[i],conf);
      if (conf>ThrC)
        ThrC=conf; 
	}
	NSampleT.clear();
}
//----------------------------------------------------------------------------
void Tracker::processFrame(Mat& _frame, Rect& _objectBox,bool&_trackstu)
{
	float ctconf;
	Mat ctPattern;
	Rect objectBoxct;
	// predict
	sampleRect(_frame, _objectBox, rSearchWindow,detectBox);
	integral(_frame, imageIntegral, CV_32F);
	getFeatureValue(imageIntegral, detectBox, detectFeatureValue);
	int radioMaxIndex;
	float radioMax;
	radioClassifier(muPositive, sigmaPositive, muNegative, sigmaNegative, detectFeatureValue, radioMax, radioMaxIndex);
	objectBoxct = detectBox[radioMaxIndex];
	getPattern(_frame(objectBoxct),ctPattern);
    NNConf(ctPattern,ctconf);
	if(ctconf>ThrC)
	{
        cout<<"׷�ٳɹ�"<<endl;
		_trackstu=true;
		_objectBox=objectBoxct;
	    Learn(_frame,_objectBox);
		//cout<<"��ض���ֵ: "<<ThrC<<endl;
	    //cout<<"Ŀ����ض�: "<<ctconf<<endl;
	}
	else 
	{
		_trackstu=false;
		 cout<<"׷��ʧ��"<<endl;
	}	
}
//-------------------------------------------------------
float Tracker::bbOverlap(Rect& box1,Rect& box2)
{
  if (box1.x > box2.x+box2.width) { return 0.0; }
  if (box1.y > box2.y+box2.height) { return 0.0; }
  if (box1.x+box1.width < box2.x) { return 0.0; }
  if (box1.y+box1.height < box2.y) { return 0.0; }

  float colInt =  min(box1.x+box1.width,box2.x+box2.width) - max(box1.x, box2.x);
  float rowInt =  min(box1.y+box1.height,box2.y+box2.height) - max(box1.y,box2.y);

  float intersection = colInt * rowInt;
  float area1 = box1.width*box1.height;
  float area2 = box2.width*box2.height;
  return intersection / (area1 + area2 - intersection);
}
