#include "Utilities.h"
#include <algorithm>

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

// distance transform function
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

// trying to remove pockets so they don't count as balls 
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

// trying to cut the pocketts out of the ball mask so they don't count as balls
void BlankPockets(Mat& ball_mask, int radius = 65)
{
	// Pockets visible in the rectified image, as (x = column, y = row)
	Point pockets[3] = { Point(0, 0), Point(717, 0), Point(1433, 0) };
	for (int p = 0; p < 3; p++)
		circle(ball_mask, pockets[p], radius, Scalar(0), FILLED);
}

// hough circles function
vector<Point2f> FindBallsHough(const Mat& table, double row_scale = 0.6, double param2 = 13.0)
{
	// Grey, smoothed (to suppress cloth texture), then rows squashed so balls are roughly round
	Mat grey, blurred, squashed;
	cvtColor(table, grey, COLOR_BGR2GRAY);
	GaussianBlur(grey, blurred, Size(7, 7), 1.5);
	resize(blurred, squashed, Size(), 1.0, row_scale);

	// dp = 1 min distance between centres = 22, Canny upper threshold = 100
	// accumulator threshold = param2, radius range 16 to 24 pixels
	vector<Vec3f> circles;
	HoughCircles(squashed, circles, HOUGH_GRADIENT, 1, 22, 100, param2, 16, 24);

	vector<Point2f> centres;
	for (size_t i = 0; i < circles.size(); i++)
		centres.push_back(Point2f(circles[i][0], circles[i][1] / (float)row_scale));   // x = column, y = row
	return centres;
}

// combining both distance transform and hough circles detections
vector<Point2f> CombineDetections(const vector<Point2f>& hough, const vector<Point2f>& dt,
                                  double min_separation = 30.0, double row_scale = 0.6)
{
	vector<Point2f> combined = hough;                    // keep every Hough detection
	for (size_t i = 0; i < dt.size(); i++)
	{
		bool covered = false;
		for (size_t j = 0; j < hough.size(); j++)
		{
			double dx = dt[i].x - hough[j].x;
			double dy = (dt[i].y - hough[j].y) * row_scale;   // measure where balls are roughly round
			if (sqrt(dx * dx + dy * dy) < min_separation) covered = true;
		}
		if (!covered) combined.push_back(dt[i]);            // distance-transform dot with no Hough dot nearby
	}
	return combined;
}

void DrawBalls(Mat& image, const vector<Point2f>& centres, Scalar colour)
{
	for (size_t i = 0; i < centres.size(); i++)
{
	Point p(cvRound(centres[i].x), cvRound(centres[i].y));
	ellipse(image, p, Size(21, 38), 0, 0, 360, Scalar(255, 255, 255), 2);   // white ring for expected outline
	circle(image, p, 5, colour, FILLED);	// expected center 
}
}

// evaluation logic 
const int ground_truth[7][16][2] = {
	{ // AllBalls.jpeg
		{253, 908}, {250, 959}, {255, 1013}, {256, 1072}, {254, 1132}, {262, 1184}, {256, 1244}, {253, 1307},
		{329, 901}, {330, 961}, {334, 1130}, {335, 1013}, {336, 1067}, {342, 1245}, {345, 1187}, {293, 797} },
	{ // Table1.jpeg
		{309, 1092}, {429, 1212},  {382, 1133}, {335, 1050}, {285, 1211}, {261, 1170}, {352, 1173}, {331, 1132},
		{240, 1210}, {310, 1170}, {407, 1169}, {281, 1135}, {374, 1212}, {360, 1091}, {331, 1213}, {288, 29} }, // added {382, 1133} for the missing ball
	{ // Table2.jpeg
		{282, 1103}, {620, 1224}, {391, 1121}, {310, 1305}, {315, 1046}, {202, 621}, {361, 1173}, {331, 1132},
		{177, 725}, {185, 1269}, {665, 798}, {296, 438}, {53, 1235}, {362, 1090}, {399, 1205}, {507, 1169} },
	{ // Table3.jpeg
		{23, 908}, {616, 1224}, {447, 1199}, {336, 29}, {305, 1406}, {210, 621}, {260, 1294}, {226, 1252},
		{177, 725}, {171, 1270}, {665, 798}, {249, 351}, {53, 1233}, {169, 901}, {399, 1201}, {434, 29} },
	{ // Table4.jpeg
		{23, 908}, {546, 1407}, {444, 1204}, {-1, -1}, {305, 1406}, {210, 621}, {260, 1294}, {226, 1252},
		{181, 724}, {171, 1270}, {665, 798}, {249, 351}, {58, 1231}, {169, 901}, {399, 1201}, {88, 1392} },
	{ // Table5.jpeg
		{535, 23}, {546, 1407}, {444, 1204}, {-1, -1}, {305, 1406}, {131, 25}, {260, 1294}, {226, 1252},
		{538, 72}, {171, 1270}, {665, 798}, {249, 351}, {-1, -1}, {169, 901}, {399, 1201}, {608, 26} },
	{ // Table6.jpeg
		{535, 23}, {477, 1364}, {444, 1204}, {-1, -1}, {305, 1406}, {131, 25}, {260, 1294}, {226, 1252},
		{538, 72}, {171, 1270}, {-1, -1}, {249, 351}, {-1, -1}, {169, 901}, {399, 1201}, {629, 947} },
};

struct DetectionScore
{
	int true_positives = 0;
	int false_positives = 0;
	int false_negatives = 0;
	double error_sum = 0.0;      // sum of distances in mm of matched detections
};

// Compare detections (x = column, y = row) with the ground truth (row, column) for one image
DetectionScore ScoreDetections(const vector<Point2f>& detections, int image_index, double tolerance = 21.5)
{
	struct Candidate { double distance; int truth; int detection; };
	vector<Candidate> candidates;
	int truth_count = 0;
	for (int b = 0; b < 16; b++)
	{
		if (ground_truth[image_index][b][0] < 0) continue;       // ball not on the table
		truth_count++;
		for (size_t d = 0; d < detections.size(); d++)
		{
			double dx = detections[d].x - ground_truth[image_index][b][1];
			double dy = detections[d].y - ground_truth[image_index][b][0];
			double distance = sqrt(dx * dx + dy * dy);
			if (distance <= tolerance)
			{
				Candidate c = { distance, b, (int)d };
				candidates.push_back(c);
			}
		}
	}
	// One-to-one matching, closest pairs first
	sort(candidates.begin(), candidates.end(),
	     [](const Candidate& a, const Candidate& b) { return a.distance < b.distance; });
	vector<bool> truth_used(16, false), detection_used(detections.size(), false);
	DetectionScore score;
	for (size_t i = 0; i < candidates.size(); i++)
	{
		if (truth_used[candidates[i].truth] || detection_used[candidates[i].detection]) continue;
		truth_used[candidates[i].truth] = true;
		detection_used[candidates[i].detection] = true;
		score.true_positives++;
		score.error_sum += candidates[i].distance;
	}
	score.false_positives = (int)detections.size() - score.true_positives;
	score.false_negatives = truth_count - score.true_positives;
	return score;
}

void PrintScore(const string& name, const DetectionScore& s)
{
	double tp = s.true_positives, fp = s.false_positives, fn = s.false_negatives;
	double precision = (tp + fp > 0) ? tp / (tp + fp) : 0.0;
	double recall    = (tp + fn > 0) ? tp / (tp + fn) : 0.0;
	double f1        = (precision + recall > 0) ? 2 * precision * recall / (precision + recall) : 0.0;
	double mean_error = (tp > 0) ? s.error_sum / tp : 0.0;
	cout << name << ": TP " << s.true_positives << " FP " << s.false_positives << " FN " << s.false_negatives
	     << "  precision " << precision << "  recall " << recall << "  F1 " << f1
	     << "  mean error " << mean_error << " mm" << endl;
}




void DrawGroundTruth(Mat& image, int image_index)
{
	for (int b = 0; b < 16; b++)
	{
		if (ground_truth[image_index][b][0] < 0) continue;     // ball not on the table
		Point centre(ground_truth[image_index][b][1], ground_truth[image_index][b][0]);
		drawMarker(image, centre, Scalar(255, 255, 255), MARKER_CROSS, 34, 6);   // white outline
		drawMarker(image, centre, Scalar(0, 0, 0), MARKER_CROSS, 30, 3);         // black cross on top
	}
}

void AddScore(DetectionScore& total, const DetectionScore& s)
{
   total.true_positives  += s.true_positives;
   total.false_positives += s.false_positives;
   total.false_negatives += s.false_negatives;
   total.error_sum       += s.error_sum;
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

	DetectionScore total;
	DetectionScore total_dt, total_hough;

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
		BlankPockets(ball_mask);

		//1: distance transform
		vector<Point2f> dt_balls = FindBallsDT(ball_mask);
		RemovePocketDetections(dt_balls);
		Mat show_dt = table.clone();
		DrawBalls(show_dt, dt_balls, Scalar(0, 0, 255));          // red

		//2: Hough circles
		vector<Point2f> hough_balls = FindBallsHough(table);
		RemovePocketDetections(hough_balls);
		Mat show_hough = table.clone();
		DrawBalls(show_hough, hough_balls, Scalar(255, 0, 0));    // blue

		// 3: combining the two detections 
		vector<Point2f> final_balls = CombineDetections(hough_balls, dt_balls);
		Mat show_final = table.clone();
		DrawBalls(show_final, final_balls, Scalar(0, 200, 0));    // green
		
		cout << pool_files[file_no] << ": DT " << dt_balls.size()
    		 << ", Hough " << hough_balls.size()
     		 << ", combined " << final_balls.size() << endl;

		DetectionScore score = ScoreDetections(final_balls, file_no);
		PrintScore(pool_files[file_no], score);
				
		if (file_no > 0)          // pooled totals use Table1 to Table6 only
		{
		    AddScore(total,       score);
		    AddScore(total_dt,    ScoreDetections(dt_balls, file_no));
		    AddScore(total_hough, ScoreDetections(hough_balls, file_no));
		}

		DrawGroundTruth(show_final, file_no);

		// imshow(string(pool_files[file_no]) + " - DT (red)", show_dt);
		// imshow(string(pool_files[file_no]) + " - Hough (blue)", show_hough);

		imshow(string(pool_files[file_no]) + " - combined (green)", show_final);
		

		waitKey();

		// destroyWindow(string(pool_files[file_no]) + " - DT (red)");
		// destroyWindow(string(pool_files[file_no]) + " - Hough (blue)");
		destroyWindow(string(pool_files[file_no]) + " - combined (green)");

		imwrite(string("out_") + pool_files[file_no] + "_step2_background.png", background);
		imwrite(string("out_") + pool_files[file_no] + "_step3_ball_mask.png", ball_mask);
		imwrite(string("out_") + pool_files[file_no] + "_step4_dt.png", show_dt);
		imwrite(string("out_") + pool_files[file_no] + "_step5_hough.png", show_hough);
		imwrite(string("out_") + pool_files[file_no] + "_step6_combined.png", show_final);
	}


	PrintScore("TOTAL Table1-6", total);
   	PrintScore("TOTAL DT only", total_dt);
  	PrintScore("TOTAL Hough only", total_hough);
}
	
