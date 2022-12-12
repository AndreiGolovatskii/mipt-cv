#include <assert.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <opencv2/core/base.hpp>
#include <opencv2/core/cvstd.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <string>
#include <string_view>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc/fast_hough_transform.hpp>

std::size_t next_power2(std::size_t value) {
  return 1u << (32 - __builtin_clz(value - 1));
}

std::pair<std::size_t, std::size_t> borders(std::size_t origin_size, std::size_t target_size) {
  assert(target_size >= origin_size);
  auto l = (target_size - origin_size) / 2;
  return {l, (target_size - origin_size) - l};
}

cv::Mat fill_borders(cv::Mat img) {
  auto target_size = std::max(next_power2(img.size[0]), next_power2(img.size[1]));

  auto [top, bottom] = borders(img.size[0], target_size);
  auto [left, right] = borders(img.size[1], target_size);

  auto out = cv::Mat(target_size, target_size, CV_8U);
  cv::copyMakeBorder(img, out, top, bottom, left, right, cv::BORDER_CONSTANT);

  return out;
}

cv::Mat fht_impl(cv::Mat img, std::size_t W, std::size_t xmin, std::size_t xmax) {
  cv::Mat res(W, xmax - xmin, CV_32FC1);
  if (xmax - xmin == 1) {
    img.col(xmin).copyTo(res.col(0));
  } else {
    std::size_t mid = (xmin + xmax) / 2;
    auto a1 = fht_impl(img, W, xmin, mid);
    auto a2 = fht_impl(img, W, mid, xmax);

    for (std::size_t x = 0; x < W; ++x) {
      for (std::size_t t = 0; t < xmax - xmin; ++t) {
        std::size_t t0 = t / 2;
        std::size_t shift = t - t0;
        if (x < (W - shift)) {
          res.at<float>(x, t) = a1.at<float>(x, t0) + a2.at<float>(x + shift, t0);
        } else {
          res.at<float>(x, t) = a1.at<float>(x, t0) + a2.at<float>(x - (W - shift), t0);
        }
      }
    }
  }

  res.at<float>(0, 0) = 0;
  cv::Mat normalized(W, xmax - xmin, CV_32FC1);
  cv::normalize(res, normalized, 0, 1, cv::NORM_MINMAX);
  return normalized;
}

cv::Mat fht(cv::Mat img) {
  return fht_impl(img, img.size[0], 0, img.size[1]);
}

cv::Mat grayscale(cv::Mat img) {
  auto gray = cv::Mat(img.size[0], img.size[1], CV_32FC1);
  cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
  return gray;
}

cv::Mat canny(cv::Mat img) {
  auto kernel = cv::getStructuringElement(cv::MORPH_RECT, {1, 1});

  auto morph = cv::Mat(img.size[0], img.size[1], img.type());
  cv::morphologyEx(img, morph, cv::MORPH_CLOSE, kernel, cv::Point(-1, -1), 1);

  auto canny = cv::Mat(img.size[0], img.size[1], img.type());
  cv::Canny(morph, canny, 0, 200);
  return canny;
}

int get_rotation_angle(cv::Mat fht, int interpolation) {
  auto interpolated = cv::Mat(180, 180, CV_16FC1);
  cv::resize(fht, interpolated, interpolated.size(), 0, 0, interpolation);
  assert(interpolated.size[0] == 180);

  std::size_t max_x = 0;
  float max_v = 0;
  for (int x = 0; x < interpolated.size[0]; ++x) {
    cv::Mat mean;
    cv::Mat stddev;

    cv::meanStdDev(interpolated.col(x), mean, stddev);

    if (max_v < stddev.at<double>(0)) {
      std::tie(max_x, max_v) = std::tie(x, stddev.at<double>(0));
    }
  }
  int angle = (int(atan2(interpolated.size[0], max_x) * 180 / M_PI) % 90 + 90) % 90;
  if (angle > 45) {
    angle -= 90;
  }
  return angle;
}

cv::Mat rotate(cv::Mat img, int angle, int interpolation) {
  auto image_center = cv::Point2f((img.size[1] - 1) / 2., (img.size[0] - 1) / 2.);
  auto mtx = cv::getRotationMatrix2D(image_center, -angle, 1.0);

  cv::Rect2f bbox = cv::RotatedRect(cv::Point2f(), img.size(), -angle).boundingRect2f();
  mtx.at<double>(0, 2) += bbox.width / 2.0 - img.cols / 2.0;
  mtx.at<double>(1, 2) += bbox.height / 2.0 - img.rows / 2.0;

  cv::Mat rotated;
  cv::warpAffine(img, rotated, mtx, bbox.size(), interpolation);
  return rotated;
}

struct timer_guard {
  explicit timer_guard()
    : start(std::chrono::high_resolution_clock::now()) {}

  ~timer_guard() {
    std::cout << (std::chrono::high_resolution_clock::now() - start).count() << '\n';
  }

  std::chrono::high_resolution_clock::time_point start;
};

int main(int argc, char *argv[]) {
  assert(argc == 3);
  char *im_path = argv[1];
  bool use_linear = [&] {
    using namespace std::literals;
    if (argv[2] == "bilinear"sv) {
      return true;
    } else if (argv[2] == "nearest"sv) {
      return false;
    }
    assert(false && "bilinear or nearest argument expected");
  }();

  cv::Mat img = cv::imread(im_path + std::string(".jpg"), cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cout << "Could not read the image: " << std::string(im_path) << std::endl;
    return 1;
  }

  auto out = [&] {
    timer_guard timer;
    auto gray = grayscale(img);
    auto canny_img = canny(gray);

    auto bordered = fill_borders(canny_img);

    cv::imwrite(std::string("precised.jpg"), bordered);
    auto angle = get_rotation_angle(fht(bordered), use_linear ? cv::INTER_LINEAR : cv::INTER_NEAREST);

    return rotate(img, angle, use_linear ? cv::INTER_LINEAR: cv::INTER_NEAREST);
  }();

  cv::imwrite(im_path + std::string("-out.jpg"), out);
}
