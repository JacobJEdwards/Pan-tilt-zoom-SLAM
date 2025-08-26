#define CVUI_IMPLEMENTATION
#include "annotation_window.h"

AnnotationWindow::AnnotationWindow(cv::String name, std::string index)
{
	main_view_name_ = name;

	frame_ = cv::Mat(200, 400, CV_8UC3);
	frame_ = cv::Scalar(50, 50, 50);

	state_ = AnnotationState::point;

	visualize_view_ = new VisualizeView("Visualize View");

	img_index_ = index;

#ifdef _WIN32
	io_util::readModel("./resource/ice_hockey_model.txt", points_, pairs_);
#elif __APPLE__
	io_util::readModel("/Users/jacobedwards/dev/Pan-tilt-zoom-SLAM/gui/resource/ice_hockey_model.txt", points_, pairs_);
#endif
}

AnnotationWindow::~AnnotationWindow() {}

void AnnotationWindow::setFeatureAnnotationView(FeatureAnnotationView* image_view)
{
	feature_annotation_view_ = image_view;
}

void AnnotationWindow::setCourtView(CourtView* image_view)
{
	court_view_ = image_view;
}

void AnnotationWindow::visualize_camera(vpgl_perspective_camera<double> camera)
{
	cv::Mat img = feature_annotation_view_->getImage().clone();

	for (auto iter = pairs_.begin(); iter != pairs_.end(); ++iter)
	{
		int p1_index = iter->first;
		int p2_index = iter->second;

		vgl_point_2d<double> p1 = points_[p1_index];
		vgl_point_2d<double> p2 = points_[p2_index];

		vgl_homg_point_3d<double> p1_3d(p1.x(), p1.y(), 0);
		vgl_homg_point_3d<double> p2_3d(p2.x(), p2.y(), 0);

		vgl_homg_point_2d<double> result1 = camera.project(p1_3d);
		vgl_homg_point_2d<double> result2 = camera.project(p2_3d);

		cv::line(img, cv::Point(result1.x() / result1.w(), result1.y() / result1.w()),
			cv::Point(result2.x() / result2.w(), result2.y() / result2.w()), cv::Scalar(255, 0, 0), 3);
	}
	cv::imwrite("visualization_debug.jpg", img);

	//cv::imshow("test", img);
	//cv::waitKey(0);
	visualize_view_->setImage(img);
}

void AnnotationWindow::calibButtonFunc()
{
	vector<vgl_point_2d<double>> points_annotation = feature_annotation_view_->getPoints();
	vector<vgl_point_2d<double>> points_court = court_view_->getPoints();

	printf("Annotations in image:\n");
	for (auto iter = points_annotation.begin(); iter != points_annotation.end(); ++iter)
	{
		printf("(%f, %f)\n", iter->x(), iter->y());
	}

	printf("points in world coordinates:\n");
	for (auto iter = points_court.begin(); iter != points_court.end(); ++iter)
	{
		printf("(%f, %f)\n", iter->x(), iter->y());
	}

	if (points_annotation.size() == points_court.size() &&
		points_annotation.size() >= 4)
	{
		vgl_point_2d<double> principal_point(640, 360);
		//vpgl_perspective_camera<double> camera;

		bool is_calib = cvx::init_calib(points_court, points_annotation, principal_point, init_camera_);
		if (is_calib)
		{
			printf("successfully init calib.\n");
			if (refineCalibIceHockey())
			{
				visualize_camera(opt_camera_);

				io_util::writeCamera("camera.txt", (img_index_ + ".jpg").c_str(), opt_camera_);
			}
			else
			{
				visualize_camera(init_camera_);
				io_util::writeCamera("camera.txt", (img_index_ + ".jpg").c_str(), init_camera_);
			}



			//vpgl_perspective_camera<double> refined_camera;
			//bool is_optimized = cvx::optimize_perspective_camera(points_court, points_annotation,
			//	camera, refined_camera);
			//if (is_optimized) {
			//	camera = refined_camera;
			//}
			// draw annotation
			// save camera

			/*cv::Mat img = cv::Mat(feature_annotation_view_->getImage());

			for (auto iter = pairs_.begin(); iter != pairs_.end(); ++iter)
			{
				int p1_index = iter->first;
				int p2_index = iter->second;

				vgl_point_2d<double> p1 = points_[p1_index];
				vgl_point_2d<double> p2 = points_[p2_index];

				vgl_homg_point_3d<double> p1_3d(p1.x(), p1.y(), 0);
				vgl_homg_point_3d<double> p2_3d(p2.x(), p2.y(), 0);


				vgl_homg_point_2d<double> result1 = opt_camera_.project(p1_3d);
				vgl_homg_point_2d<double> result2 = opt_camera_.project(p2_3d);

				cv::line(img, cv::Point(result1.x() / result1.w(), result1.y() / result1.w()),
					cv::Point(result2.x() / result2.w(), result2.y() / result2.w()), cv::Scalar(255, 0, 0), 3);
			}
			cv::imwrite("visualization_debug.jpg", img);

			visualize_view_->setImage(img);*/
		}
		else
		{
			printf("Warning: init calib failed.\n");
		}

	}
}

bool AnnotationWindow::refineCalibIceHockey()
{
	//get point from image and world
	vector<vgl_point_2d<double> > world_pts = court_view_->getPoints(); // = [m_courtImageView getPoints:true];
	vector<vgl_point_2d<double> > image_pts = feature_annotation_view_->getPoints(); // @1 This the point in image  = [m_orgImageView pts_];
	if (!(world_pts.size() >= 2 && image_pts.size() >= 2)) {
		printf("Warning: Ice hockey, at least two point correspondences.\n");
		return false;
	}
	if (world_pts.size() != image_pts.size()) {
		printf("Warning: Ice hockey, number of world points and image points is not equal.\n");
		return false;
	}

	// line segments
	vector<vgl_line_3d_2_points<double> > world_line;
	vector<vgl_line_segment_2d<double> > world_line_segment = court_view_->getLines(); // = [m_courtImageView getLineSegment: true];
	for (int i = 0; i < world_line_segment.size(); i++) {
		vgl_point_3d<double> p1(world_line_segment[i].point1().x(), world_line_segment[i].point1().y(), 0);
		vgl_point_3d<double> p2(world_line_segment[i].point2().x(), world_line_segment[i].point2().y(), 0);
		world_line.push_back(vgl_line_3d_2_points<double>(p1, p2));
	}

	vector<vgl_line_segment_2d<double> > image_line_segment = feature_annotation_view_->getLines();// @2 This is the line in imge = [m_orgImageView lines_];


	// circle

	// group circle annotation into five circles
	vector<vgl_conic<double>> world_conics = NHLIceHockeyPlayField::getCircles();
	vector<vgl_point_2d<double>> circle_pts = feature_annotation_view_->getCirclePoints(); // @3 this point in image // = [m_orgImageView circle_pts_];

	if (image_line_segment.size() == world_line.size() && (world_line.size() > 0 || circle_pts.size() > 0))
	{
		vector<vector<vgl_point_2d<double> > > image_line_pt_groups;
		for (int i = 0; i < image_line_segment.size(); i++) {
			vector<vgl_point_2d<double> > pts;
			pts.push_back(image_line_segment[i].point1());
			pts.push_back(image_line_segment[i].point2());
			pts.push_back(centre(pts[0], pts[1]));
			image_line_pt_groups.push_back(pts);
		}       
       
        io_util::writeGUIRecord("debug.txt", "debug.jpg", world_pts, image_pts,
                                world_line, image_line_segment,
                                world_conics, circle_pts);
        printf("save file to debug.txt\n");
        

		bool is_opt = cvx::optimize_perspective_camera_point_line_circle(world_pts, image_pts,
			world_line, image_line_pt_groups,
			world_conics, circle_pts, init_camera_, opt_camera_);
		if (is_opt) {
			//[self drawCourtAndSave:opt_camera];
			printf("Ice hockey camera refinement done.\n");
			return true;
		}
		else {
			printf("Warning: refine camera failed\n");
			return false;
		}
	}

	return false;
}

void AnnotationWindow::clearButtonFunc()
{
	feature_annotation_view_->clearAnnotations();
	court_view_->clearAnnotations();
}

void AnnotationWindow::mainControlHandler()
{
	if (cvui::button(frame_, 100, 30, "Do Calibration")) {
		calibButtonFunc();
	}

	if (cvui::button(frame_, 100, 60, "Clear Annotations")) {
		clearButtonFunc();
	}

	annotationStateFunc();
}

void AnnotationWindow::annotationStateFunc()
{
	bool point_checkbox = false;
	bool line_checkbox = false;
	bool circle_checkbox = false;

	switch (state_)
	{
	case AnnotationState::point:
		point_checkbox = true;
		break;
	case AnnotationState::line:
		line_checkbox = true;
		break;
	case AnnotationState::circle:
		circle_checkbox = true;
		break;
	}

	cvui::checkbox(frame_, 100, 100, "point", &point_checkbox);
	cvui::checkbox(frame_, 100, 120, "line", &line_checkbox);
	cvui::checkbox(frame_, 100, 140, "circle", &circle_checkbox);

	switch (state_)
	{
	case AnnotationState::point:
		if (line_checkbox)
			state_ = AnnotationState::line;
		else if (circle_checkbox)
			state_ = AnnotationState::circle;
		break;
	case AnnotationState::line:
		if (point_checkbox)
			state_ = AnnotationState::point;
		else if (circle_checkbox)
			state_ = AnnotationState::circle;
		break;
	case AnnotationState::circle:
		if (point_checkbox)
			state_ = AnnotationState::point;
		else if (line_checkbox)
			state_ = AnnotationState::line;
		break;
	}

	court_view_->setState(state_);
	feature_annotation_view_->setState(state_);
}

void AnnotationWindow::startLoop() {

	const int view_number = 4;

	const cv::String feature_view_name = feature_annotation_view_->getWindowName();
	const cv::String court_view_name = court_view_->getWindowName();
	const cv::String visualize_view_name = visualize_view_->getWindowName();

	const cv::String view_names[] =
	{
		feature_view_name,
		court_view_name,
		main_view_name_,
		visualize_view_name
	};

	// init multiple windows
	cvui::init(view_names, view_number);

	while (true) {
		frame_ = cv::Scalar(50, 50, 50);

		cvui::context(feature_view_name);
		feature_annotation_view_->drawFrame();
		feature_annotation_view_->annotate();
		cvui::imshow(feature_view_name, feature_annotation_view_->frame_);

		cvui::context(court_view_name);
		court_view_->drawFrame();
		court_view_->annotate();
		cvui::imshow(court_view_name, court_view_->frame_);

		cvui::context(main_view_name_);
		mainControlHandler();
		cvui::imshow(main_view_name_, frame_);

		cvui::context(visualize_view_name);
		visualize_view_->drawFrame();
		cvui::imshow(visualize_view_name, visualize_view_->frame_);

		// Check if ESC key was pressed
		if (cv::waitKey(20) == 27) {
			break;
		}
	}
}

