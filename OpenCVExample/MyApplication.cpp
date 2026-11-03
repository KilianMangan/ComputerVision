#include "Utilities.h"

// Step 1: warp the camera image to the top-down table view (683 rows x 1434 cols)
Mat WarpToTable(const Mat& src)
{
	// Table corners in the original image as (x = column, y = row): TL, BL, TR, BR
	Point2f from[4] = { Point2f(337, 35), Point2f(38, 472), Point2f(1570, 21), Point2f(1888, 451) };
	Point2f to[4]   = { Point2f(0, 0),    Point2f(0, 682),  Point2f(1433, 0),  Point2f(1433, 682) };

	Mat H = getPerspectiveTransform(from, to);
	Mat table;
	warpPerspective(src, table, H, Size(1434, 683));
	return table;
}


// step 2: making the table cloth mask
Mat BackgroundMask(const Mat& table)
{
	Mat blurred, hls;
	GaussianBlur(table, blurred, Size(5, 5), 0);
	cvtColor(blurred, hls, COLOR_BGR2HLS);
	vector<Mat> ch;
	split(hls, ch);              // ch[0]=H, ch[1]=L, ch[2]=S

	// Cloth is the dominant colour, so its hue is the peak of the hue histogram
	int bins = 180; float range[] = { 0, 180 }; const float* ranges = range; int channel = 0;
	Mat hist;
	calcHist(&ch[0], 1, &channel, Mat(), hist, 1, &bins, &ranges);
	Point peak;
	minMaxLoc(hist, NULL, NULL, NULL, &peak);
	int cloth_hue = peak.y;
	cout << "Cloth hue: " << cloth_hue << endl;


	Mat hue_diff;
	absdiff(ch[0], Scalar(cloth_hue), hue_diff);
	Mat hue_ok = hue_diff <= 10;
	Mat sat_ok = ch[2] >= 20;
	Mat cloth;
	bitwise_and(hue_ok, sat_ok, cloth);

	// brownish frame
	Mat d44;
	absdiff(ch[0], Scalar(44), d44);
	Mat frame_hue = d44 <= 8;
	Mat dark = ch[1] < 90;
	Mat frame;
	bitwise_and(frame_hue, dark, frame);

	// Cushion: green-ish but more saturated than the cloth
	Mat cush_hue = hue_diff <= 16;
	Mat cush_sat = ch[2] >= 60;
	Mat cushion;
	bitwise_and(cush_hue, cush_sat, cushion);

	Mat background = cloth.clone();
	bitwise_or(background, frame, background);
	bitwise_or(background, cushion, background);
	return background;
}

// masking the balls, step  4
Mat BallMask(const Mat& background)
{
	Mat k15 = getStructuringElement(MORPH_ELLIPSE, Size(15, 15));
	Mat k7  = getStructuringElement(MORPH_ELLIPSE, Size(7, 7));

	// Playing area = convex hull of the largest opened background region
	Mat opened;
	morphologyEx(background, opened, MORPH_OPEN, k15);
	vector<vector<Point>> contours;
	findContours(opened, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	int largest = 0; double largest_area = 0.0;
	for (int c = 0; c < (int)contours.size(); c++)
	{
		double a = contourArea(contours[c]);
		if (a > largest_area) { largest_area = a; largest = c; }
	}
	vector<vector<Point>> hulls(1);
	convexHull(contours[largest], hulls[0]);
	Mat play = Mat::zeros(background.size(), CV_8U);
	drawContours(play, hulls, 0, Scalar(255), FILLED);

	// Balls = not background, inside the playing area, cleaned up
	Mat not_background, balls;
	bitwise_not(background, not_background);
	bitwise_and(not_background, play, balls);
	morphologyEx(balls, balls, MORPH_OPEN, k7);
	morphologyEx(balls, balls, MORPH_CLOSE, k15);
	return balls;
}

vector<Point2f> FindBallsDT(const Mat& ball_mask, double row_scale = 0.6,
                            double min_radius = 12.0, int window = 21)
{
	// Balls look about 43 px wide but about 76 px tall after rectification, so squash rows
	Mat squashed;
	resize(ball_mask, squashed, Size(), 1.0, row_scale, INTER_NEAREST);

	Mat dt;
	distanceTransform(squashed, dt, DIST_L2, 5);

	// Local maxima: pixels equal to the maximum in their neighbourhood
	Mat dilated;
	dilate(dt, dilated, getStructuringElement(MORPH_ELLIPSE, Size(window, window)));
	Mat is_max = dt >= dilated;
	Mat big_enough = dt >= min_radius;
	Mat peaks;
	bitwise_and(is_max, big_enough, peaks);

	// One centre per peak region, converted back to table coordinates
	vector<vector<Point>> contours;
	findContours(peaks, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
	vector<Point2f> centres;
	for (int c = 0; c < (int)contours.size(); c++)
	{
		Rect r = boundingRect(contours[c]);
		centres.push_back(Point2f(r.x + r.width / 2.0f,
		                          (r.y + r.height / 2.0f) / (float)row_scale));   // x = column, y = row
	}
	return centres;
}

// trying to remove pockets so they don't count as ablls 
void RemovePocketDetections(vector<Point2f>& centres, double pocket_radius = 85.0)
{
	// Pockets visible in the rectified image, as (x = column, y = row)
	Point2f pockets[3] = { Point2f(0, 0), Point2f(717, 0), Point2f(1433, 0) };
	vector<Point2f> kept;
	for (size_t i = 0; i < centres.size(); i++)
	{
		bool near_pocket = false;
		for (int p = 0; p < 3; p++)
		{
			double dx = centres[i].x - pockets[p].x;
			double dy = centres[i].y - pockets[p].y;
			if (sqrt(dx * dx + dy * dy) < pocket_radius) near_pocket = true;
		}
		if (!near_pocket) kept.push_back(centres[i]);
	}
	centres = kept;
}

void DrawBalls(Mat& image, const vector<Point2f>& centres, Scalar colour)
{
	for (size_t i = 0; i < centres.size(); i++)
{
	Point p(cvRound(centres[i].x), cvRound(centres[i].y));
	circle(image, p, 10, Scalar(255, 255, 255), 2);   // white ring
	circle(image, p, 5, colour, FILLED);
}
}

void MyApplication()
{
	const char* file_location = "Media/Pool/";
	const char* pool_files[] = {
		"AllBalls.jpeg",
		"Table1.jpeg",
		"Table2.jpeg",
		"Table3.jpeg",
		"Table4.jpeg",
		"Table5.jpeg",
		"Table6.jpeg"
	};
	int number_of_images = sizeof(pool_files) / sizeof(pool_files[0]);

	for (int file_no = 0; file_no < number_of_images; file_no++)
	{
		string filename(file_location);
		filename.append(pool_files[file_no]);
		Mat image = imread(filename, -1);
		if (image.empty())
		{
			cout << "Could not open " << filename << endl;
			continue;
		}

		Mat table = WarpToTable(image);
		Mat background = BackgroundMask(table);
		Mat ball_mask = BallMask(background);

		vector<Point2f> dt_balls = FindBallsDT(ball_mask);
		RemovePocketDetections(dt_balls);

		Mat show_dt = table.clone();
		DrawBalls(show_dt, dt_balls, Scalar(0, 0, 255));
		cout << pool_files[file_no] << ": " << dt_balls.size() << " balls found" << endl;
		imshow(string(pool_files[file_no]) + " - DT detections", show_dt);
		imwrite(string("out_") + pool_files[file_no] + "_step4_dt.png", show_dt);


		
		waitKey();;

		destroyWindow(string(pool_files[file_no]) + " - DT detections");

		imwrite(string("out_") + pool_files[file_no] + "_step2_background.png", background);
		imwrite(string("out_") + pool_files[file_no] + "_step3_ball_mask.png", ball_mask);
		imwrite(string("out_") + pool_files[file_no] + "_step4_dt.png", show_dt);
	}
}