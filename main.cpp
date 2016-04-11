#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "objdetect.hpp"
#include <stdio.h>
#include <string>
using namespace std;
using namespace cv;

Rect BoundRect;
String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;

static void help()
{
	printf("\n"
		"This program demonstrated a simple method of connected components clean up of background subtraction\n"
		"When the program starts, it begins learning the background.\n"
		"You can toggle background learning on and off by hitting the space bar.\n"
		"Call\n"
		"./segment_objects [video file, else it reads camera 0]\n\n");
}
static void refineSegments(const Mat& img, Mat& mask, Mat& dst)
{
	int niters = 1;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat temp;
	dilate(mask, temp, Mat(), Point(-1, -1), niters);
	erode(temp, temp, Mat(), Point(-1, -1), niters * 2);
	dilate(temp, temp, Mat(), Point(-1, -1), niters);
	findContours(temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
	dst = Mat::zeros(img.size(), CV_8UC3);
	if (contours.size() == 0)
		return;
	// iterate through all the top-level contours,
	// draw each connected component with its own random color
	int idx = 0, largestComp = 0;
	double maxArea = 0;
	for (; idx >= 0; idx = hierarchy[idx][0])
	{
		const vector<Point>& c = contours[idx];
		double area = fabs(contourArea(Mat(c)));
		if (area > maxArea)
		{
			maxArea = area;
			largestComp = idx;
		}
	}
	Scalar color(0, 0, 255);
	drawContours(dst, contours, largestComp, color, FILLED, LINE_8, hierarchy);

	BoundRect = boundingRect(contours[largestComp]);
	rectangle(dst, BoundRect, (0, 0, 255));
}
int main(int argc, char** argv)
{
	VideoCapture cap;
	bool update_bg_model = true;
	CommandLineParser parser(argc, argv, "{help h||}{@input||}");
	if (parser.has("help"))
	{
		help();
		return 0;
	}
	string input = parser.get<std::string>("@input");
	if (input.empty())
		cap.open(0);
	else
		cap.open(input);
	if (!cap.isOpened())
	{
		printf("\nCan not open camera or video file\n");
		return -1;
	}
	Mat tmp_frame, bgmask, out_frame;
	cap >> tmp_frame;
	if (tmp_frame.empty())
	{
		printf("can not read data from the video source\n");
		return -1;
	}
	if (!face_cascade.load(face_cascade_name))
	{ 
		printf("--(!)Error loading\n"); return -2;
	};
	namedWindow("video", 1);
	namedWindow("segmented", 1);
	Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
	bgsubtractor->setVarThreshold(16);
	for (;;)
	{
		cap >> tmp_frame;
		if (tmp_frame.empty())
			break;
		bgsubtractor->apply(tmp_frame, bgmask, update_bg_model ? -1 : 0);
		refineSegments(tmp_frame, bgmask, out_frame);
		rectangle(tmp_frame, BoundRect, Scalar(0, 0, 255));





		vector<Rect> faces;
		Mat frame_gray = tmp_frame(BoundRect);

		cvtColor(frame_gray, frame_gray, CV_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		for (size_t i = 0; i < faces.size(); i++)
		{
			Point center(faces[i].x + faces[i].width / 2 + BoundRect.x, faces[i].y + faces[i].height / 2 + BoundRect.y);
			printf("Found a face at (%d, %d)\n", center.x, center.y);
			ellipse(tmp_frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 255, 255), 2, 8, 0);
			//waitKey(0);
		}










		imshow("video", tmp_frame);
		imshow("segmented", out_frame);
		int keycode = waitKey(30);
		if (keycode == 27)
			break;
		if (keycode == ' ')
		{
			update_bg_model = !update_bg_model;
			printf("Learn background is in state = %d\n", update_bg_model);
		}
	}
	return 0;
}