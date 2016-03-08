#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

int main(int argc, char * argv[])
{
	if (!face_cascade.load(face_cascade_name))
	{
		cout << "haarcascade_frontalface_alt.xml failed to open!" << endl;
		return -1; 
	};

	VideoCapture capture;
	capture.open(0);

	if (!capture.isOpened())
	{
		cout << "capture device failed to open!" << endl;
		return -2;
	}
	
	Mat frame;
	int frameCount = 0;

	while (1){
		capture >> frame;

		double t = (double)cvGetTickCount();
		frameCount++;

		std::vector<Rect> faces;
		Mat frame_gray;

		cvtColor(frame, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		cout << "succeed"<<endl;
		for (size_t i = 0; i < faces.size(); i++)
		{
			cout << "1" << endl;
			Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			printf("Found a face at (%d, %d)\n", center.x, center.y);
			ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 255, 255), 2, 8, 0);
		}
		stringstream buf;
		buf << frameCount;
		string num = buf.str();
		putText(frame, num, Point(20, 30), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 3);
		imshow("hashTracker", frame);

		t = (double)cvGetTickCount() - t;
		cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;

		if (cvWaitKey(1) == 27)
			break;
		}
	return 0;
}