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
		imshow(pool_files[file_no], table);
		waitKey();
		destroyWindow(pool_files[file_no]);
	}
}