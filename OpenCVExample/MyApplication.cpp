#include "Utilities.h"

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

	// Cushion: green-ish but darker than the cloth
	Mat cush_hue = hue_diff <= 16;
	Mat cush_sat = ch[2] >= 60;
	Mat cushion;
	bitwise_and(cush_hue, cush_sat, cushion);

	Mat background = cloth.clone();
	bitwise_or(background, frame, background);
	bitwise_or(background, cushion, background);
	return background;
}



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

		string mask_window = string(pool_files[file_no]) + " - background mask";
		imshow(pool_files[file_no], table);
		imshow(mask_window, background);
		waitKey();
		destroyWindow(pool_files[file_no]);
		destroyWindow(mask_window);
		imwrite(string("out_") + pool_files[file_no] + "_step2_background.png", background);
	}
}