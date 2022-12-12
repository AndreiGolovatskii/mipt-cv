#include <assert.h>
#include <chrono>
#include <iostream>
#include <opencv2/core/base.hpp>
#include <opencv2/core/hal/interface.h>
#include <opencv2/core/matx.hpp>
#include <string>
#include <string_view>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

cv::Mat cv_median_blur(cv::Mat img, int ksize) {
  // inplace is correct
  cv::medianBlur(img, img, ksize);
  return img;
}

cv::Mat simple(cv::Mat img, int ksize) {
  assert(ksize % 2 == 1);
  auto median = [&img](int x_target, int y_target,
                       int k_half_size) -> cv::Vec3b {
    // static to prevent allocations
    static std::array<std::vector<uchar>, 3> tmp;
    for (auto &vec : tmp) {
      vec.clear();
    }

    auto x_size = img.size[0];
    auto y_size = img.size[1];

    for (int x = std::max(0, x_target - k_half_size);
         x < std::min(x_size, x_target + k_half_size + 1); ++x) {
      for (int y = std::max(0, y_target - k_half_size);
           y < std::min(y_size, y_target + k_half_size + 1); ++y) {
        const auto &pxl = img.at<cv::Vec3b>(x, y);
        tmp[0].push_back(pxl[0]);
        tmp[1].push_back(pxl[1]);
        tmp[2].push_back(pxl[2]);
      }
    }

    for (auto &vec : tmp) {
      std::sort(vec.begin(), vec.end());
    }
    auto size = tmp[0].size();
    return cv::Vec3b{tmp[0][size / 2], tmp[1][size / 2], tmp[2][size / 2]};
  };


  auto result = img.clone();
  auto x_size = result.size[0];
  auto y_size = result.size[1];

  for (int x = 0; x < x_size; ++x) {
    for (int y = 0; y < y_size; ++y) {
      result.at<cv::Vec3b>(x, y) = median(x, y, ksize / 2);
    }
  }

  return result;
}

using hist_t = std::array<std::array<uint32_t, 3>, 256>;

hist_t &operator+=(hist_t &lhs, const hist_t &rhs) {
  for (size_t i = 0; i < lhs.size(); ++i) {
    lhs[i][0] += rhs[i][0];
    lhs[i][1] += rhs[i][1];
    lhs[i][2] += rhs[i][2];
  }
  return lhs;
}

hist_t &operator-=(hist_t &lhs, const hist_t &rhs) {
  for (size_t i = 0; i < lhs.size(); ++i) {
    lhs[i][0] -= rhs[i][0];
    lhs[i][1] -= rhs[i][1];
    lhs[i][2] -= rhs[i][2];
  }
  return lhs;
}

auto get_median(const hist_t &counts, int ksize) -> cv::Vec3b {
  cv::Vec3b res;

  for (int c = 0; c < 3; ++c) {
    int rem = ksize * ksize / 2;
    for (int i = 0; i < 256; ++i) {
      rem -= counts[i][c];
      if (rem < 0) {
        res[c] = i;
        break;
      }
    }
    assert(rem < 0);
  }
  return res;
};

cv::Mat huang(cv::Mat img, int ksize) {
  assert(ksize % 2 == 1);
  int R = ksize / 2;
  auto padded = cv::Mat(img.size[0] + R, img.size[1] + R, CV_8U);
  cv::copyMakeBorder(img, padded, R, R, R, R, cv::BORDER_REPLICATE);

  auto &result = img;

  auto x_size = img.size[0];
  auto y_size = img.size[1];

  for (int x = 0; x < x_size; ++x) {
    hist_t counts({});

    for (int xi = x; xi < x + ksize; ++xi) {
      for (int yi = 0; yi < ksize; ++yi) {
        auto &pxl = padded.at<cv::Vec3b>(xi, yi);
        for (int c = 0; c < 3; ++c) {
          counts[pxl[c]][c] += 1;
        }
      }
    }

    result.at<cv::Vec3b>(x, 0) = get_median(counts, ksize);

    for (int y = 1; y < y_size; ++y) {
      for (int xi = x; xi < x + ksize; ++xi) {
        auto &pxl_old = padded.at<cv::Vec3b>(xi, y - 1);
        auto &pxl_new = padded.at<cv::Vec3b>(xi, y + ksize - 1);
        for (int c = 0; c < 3; ++c) {
          counts[pxl_old[c]][c] -= 1;
          counts[pxl_new[c]][c] += 1;
        }
      }

      result.at<cv::Vec3b>(x, y) = get_median(counts, ksize);
    }
  }
  return result;
}

cv::Mat constant_time(cv::Mat img, int ksize) {
  assert(ksize % 2 == 1);
  int R = ksize / 2;
  auto padded = cv::Mat(img.size[0] + R, img.size[1] + R, CV_8U);
  cv::copyMakeBorder(img, padded, R, R, R, R, cv::BORDER_REPLICATE);

  auto &result = img;

  auto columns_hists = std::vector<hist_t>(padded.size[1]);

  for (int y = 0; y < padded.size[1]; ++y) {
    auto &hist = columns_hists[y];
    for (int x = 0; x + 1 < ksize; ++x) {
      const auto &pxl = padded.at<cv::Vec3b>(x, y);
      for (int c = 0; c < 3; ++c) {
        hist[pxl[c]][c] += 1;
      }
    }
  }

  for (int x = 0; x < img.size[0]; ++x) {
    hist_t counts({});
    for (int xi = x; xi < x + ksize; ++xi) {
      for (int y = 0; y < ksize; ++y) {
        const auto &pxl = padded.at<cv::Vec3b>(xi, y);
        for (int c = 0; c < 3; ++c) {
          counts[pxl[c]][c] += 1;
        }
      }
    }
    result.at<cv::Vec3b>(x, 0) = get_median(counts, ksize);

    for (int y = 0; y < img.size[1] + ksize - 1; ++y) {
      if (x > 0) {
        const auto &pxl_old = padded.at<cv::Vec3b>(x - 1, y);
        for (int c = 0; c < 3; ++c) {
          columns_hists[y][pxl_old[c]][c] -= 1;
        }
      }

      const auto &pxl_new = padded.at<cv::Vec3b>(x + ksize - 1, y);
      for (int c = 0; c < 3; ++c) {
        columns_hists[y][pxl_new[c]][c] += 1;
      }

      if (y >= ksize) {
        counts += columns_hists[y];
        counts -= columns_hists[y - ksize];
        result.at<cv::Vec3b>(x, y - ksize + 1) = get_median(counts, ksize);
      }
    }
  }
  return result;
}

struct timer_guard {
  explicit timer_guard(): start(std::chrono::high_resolution_clock::now()) {}

  ~timer_guard() {
    std::cout << (std::chrono::high_resolution_clock::now() - start).count() << '\n';
  }

  std::chrono::high_resolution_clock::time_point start;
};


int main(int argc, char *argv[]) {
  assert(argc == 4);
  char *im_path = argv[1];
  int ksize = std::stoi(argv[2]);
  char *algo = argv[3];

  cv::Mat img = cv::imread(im_path + std::string(".bmp"), cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cout << "Could not read the image: " << std::string(im_path) << std::endl;
    return 1;
  }

  using namespace std::literals;

  if (algo == "cv"sv) {
    auto out = [&] {
      timer_guard timer;
      return cv_median_blur(img, ksize);
    }();

    cv::imwrite(im_path  + std::string("-cv-") + std::to_string(ksize) + ".bmp", out);
  } else if (algo == "simple"sv) {
    auto out = [&] {
      timer_guard timer;
      return simple(img, ksize);
    }();

    cv::imwrite(im_path  + std::string("-simple-") + std::to_string(ksize) + ".bmp", out);
  } else if (algo == "huang"sv) {
    auto out = [&] {
      timer_guard timer;
      return huang(img, ksize);
    }();

    cv::imwrite(im_path  + std::string("-huang-") + std::to_string(ksize) + ".bmp", out);
  } else if (algo == "const-time"sv) {
    auto out = [&] {
      timer_guard timer;
      return constant_time(img, ksize);
    }();
    cv::imwrite(im_path  + std::string("-const-time-") + std::to_string(ksize) + ".bmp", out);
  }
}
