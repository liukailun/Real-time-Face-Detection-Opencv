#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/face.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/core.hpp"

#include <stdio.h>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;
using namespace cv;
using namespace cv::face;


Rect BoundRect;
//String face_cascade_name = "haarcascade_frontalface_alt.xml";
//CascadeClassifier face_cascade;


static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char separator = ';') {
	std::ifstream file(filename.c_str(), ifstream::in);
	if (!file) {
		string error_message = "No valid input file was given, please check the given filename.";
		CV_Error(Error::StsBadArg, error_message);
	}
	string line, path, classlabel;
	while (getline(file, line)) {
		stringstream liness(line);
		getline(liness, path, separator);
		getline(liness, classlabel);
		if (!path.empty() && !classlabel.empty()) {
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));
		}
	}
}



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

	///////////////////////////////////////////////////////////////////////////////////////////////

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
		//if (area > maxArea)
		//{
		//	maxArea = area;
		//	largestComp = idx;
		//}
		if (area > 100*100)
		{
			Scalar color(0, 0, 255);
			drawContours(dst, contours, idx, color, FILLED, LINE_8, hierarchy);
			BoundRect = boundingRect(contours[idx]);
			rectangle(dst, BoundRect, color);
		}
	}
	//Scalar color(0, 0, 255);
	////drawContours(dst, contours, largestComp, color, FILLED, LINE_8, hierarchy);
	//drawContours(dst, contours, -1, color, FILLED, LINE_8, hierarchy);

	/////////////////////////////////////////////////////////////////////////////////////////////////

	//BoundRect = boundingRect(contours[largestComp]);
	//rectangle(dst, BoundRect, (0, 0, 255));
}
int main(int argc, char** argv)
{

	string fn_haar = string("haarcascade_frontalface_alt.xml");

	string fn_csv = string("photo.csv");

	int deviceId = atoi("0");

	vector<Mat> images;
	vector<int> labels;

	// Read in the data (fails if no valid input filename is given, but you'll get an error message):
	try {
		read_csv(fn_csv, images, labels);
	}
	catch (cv::Exception& e) {
		cerr << "Error opening file \"" << fn_csv << "\". Reason: " << e.msg << endl;
		// nothing more we can do
		exit(1);
	}
	// Get the height from the first image. We'll need this
	// later in code to reshape the images to their original
	// size AND we need to reshape incoming faces to this size:
	int im_width = images[0].cols;
	int im_height = images[0].rows;

	// Create a FaceRecognizer and train it on the given images:
	Ptr<BasicFaceRecognizer> model = createFisherFaceRecognizer();
	model->train(images, labels);
	// That's it for learning the Face Recognition model. You now
	// need to create the classifier for the task of Face Detection.
	// We are going to use the haar cascade you have specified in the
	// command line arguments:
	//

	//////////////////////////////////////////////////////////////////////////////
	CascadeClassifier haar_cascade;
	//haar_cascade.load(fn_haar);
	if (!haar_cascade.load(fn_haar))
	{
		printf("can not find cascade"); return -2;
	};
	// Get a handle to the Video device:
	VideoCapture cap(deviceId);
	// Check if we can use this device at all:
	if (!cap.isOpened()) {
		cerr << "Capture Device ID " << deviceId << "cannot be opened." << endl;
		return -1;
	}
	

	//VideoCapture cap;
	bool update_bg_model = true;

	//CommandLineParser parser(argc, argv, "{help h||}{@input||}");
	//if (parser.has("help"))
	//{
	//	help();
	//	return 0;
	//}
	//string input = parser.get<std::string>("@input");
	//if (input.empty())
	//	cap.open(0);
	//else
	//	cap.open(input);
	//if (!cap.isOpened())
	//{
	//	printf("\nCan not open camera or video file\n");
	//	return -1;
	//}


	Mat tmp_frame, bgmask, out_frame;
	cap >> tmp_frame;
	if (tmp_frame.empty())
	{
		printf("can not read data from the video source\n");
		return -1;
	}

	namedWindow("video", 1);
	namedWindow("segmented", 1);
	Ptr<BackgroundSubtractorMOG2> bgsubtractor = createBackgroundSubtractorMOG2();
	bgsubtractor->setVarThreshold(16);
	for (;;)
	{
		cap >> tmp_frame;
		if (tmp_frame.empty())
			break;
		//bgsubtractor->apply(tmp_frame, bgmask, update_bg_model ? -1 : 0);
		bgsubtractor->apply(tmp_frame, bgmask, update_bg_model ? -1 : 0);



		int niters = 1;
		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;
		Mat temp;
		dilate(bgmask, temp, Mat(), Point(-1, -1), niters);
		erode(temp, temp, Mat(), Point(-1, -1), niters * 2);
		dilate(temp, temp, Mat(), Point(-1, -1), niters);

		///////////////////////////////////////////////////////////////////////////////////////////////

		findContours(temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
		out_frame = Mat::zeros(tmp_frame.size(), CV_8UC3);
		if (contours.size() == 0)
			continue;
		// iterate through all the top-level contours,
		// draw each connected component with its own random color
		int idx = 0, largestComp = 0;
		double maxArea = 0;
		for (; idx >= 0; idx = hierarchy[idx][0])
		{
			const vector<Point>& c = contours[idx];
			double area = fabs(contourArea(Mat(c)));
			//if (area > maxArea)
			//{
			//	maxArea = area;
			//	largestComp = idx;
			//}
			if (area > 50 * 50)
			{
				Scalar color(0, 0, 255);
				drawContours(out_frame, contours, idx, color, FILLED, LINE_8, hierarchy);
				BoundRect = boundingRect(contours[idx]);
				rectangle(out_frame, BoundRect, color);
				rectangle(tmp_frame, BoundRect, color);

								
				Mat frame_gray = tmp_frame(BoundRect);

				cvtColor(frame_gray, frame_gray, CV_BGR2GRAY);
				equalizeHist(frame_gray, frame_gray);
				

				//vector<Rect> faces;

				//face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
				//for (size_t i = 0; i < faces.size(); i++)
				//{
				//	Point center(faces[i].x + faces[i].width / 2 + BoundRect.x, faces[i].y + faces[i].height / 2 + BoundRect.y);
				//	printf("Found a face at (%d, %d)\n", center.x, center.y);
				//	ellipse(tmp_frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 255, 255), 2, 8, 0);
				//	//waitKey(0);
				//}
				

				vector< Rect_<int> > faces;
				haar_cascade.detectMultiScale(frame_gray, faces);
				// At this point you have the position of the faces in
				// faces. Now we'll get the faces, make a prediction and
				// annotate it in the video. Cool or what?
				for (size_t i = 0; i < faces.size(); i++) {
					// Process face by face:
					Rect face_i = faces[i];
					// Crop the face from the image. So simple with OpenCV C++:
					Mat face = frame_gray(face_i);
					// Resizing the face is necessary for Eigenfaces and Fisherfaces. You can easily
					// verify this, by reading through the face recognition tutorial coming with OpenCV.
					// Resizing IS NOT NEEDED for Local Binary Patterns Histograms, so preparing the
					// input data really depends on the algorithm used.
					//
					// I strongly encourage you to play around with the algorithms. See which work best
					// in your scenario, LBPH should always be a contender for robust face recognition.
					//
					// Since I am showing the Fisherfaces algorithm here, I also show how to resize the
					// face you have just found:
					Mat face_resized;
					cv::resize(face, face_resized, Size(im_width, im_height), 1.0, 1.0, INTER_CUBIC);
					// Now perform the prediction, see how easy that is:
					int prediction = model->predict(face_resized);
					// And finally write all we've found out to the original image!
					// First of all draw a green rectangle around the detected face:

					//rectangle(tmp_frame, face_i, Scalar(0, 255, 0), 1);

					Point center(faces[i].x + faces[i].width / 2 + BoundRect.x, faces[i].y + faces[i].height / 2 + BoundRect.y);
					ellipse(tmp_frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(0, 255, 0), 2, 8, 0);
					//rectangle(tmp_frame, Point(faces[i].x + BoundRect.x, faces[i].y + BoundRect.y), Point(faces[i].x + BoundRect.x + faces[i].width, faces[i].y + BoundRect.y + faces[i].height), Scalar(0, 255, 0), 1);



					// Create the text we will annotate the box with:
					string box_text = format("Prediction = %d", prediction);
					// Calculate the position for annotated text (make sure we don't
					// put illegal values in there):
					int pos_x = std::max(face_i.tl().x - 10 + BoundRect.x, 0);
					int pos_y = std::max(face_i.tl().y - 10 + BoundRect.y, 0);
					// And now put it into the image:
					putText(tmp_frame, box_text, Point(pos_x, pos_y), FONT_HERSHEY_PLAIN, 1.0, Scalar(0, 255, 0), 2);
				}
				

			}
		}
		

		//refineSegments(tmp_frame, bgmask, out_frame);
		

		//rectangle(tmp_frame, BoundRect, Scalar(0, 0, 255));
		
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