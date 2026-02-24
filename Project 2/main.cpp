/*
* Arpit Gandhi
February 2025

Combined Image Matching System

Usage:
  build mode:  arg[1]=build arg[2]=<img_directory> arg[3]=<output_csv> arg[4]=<matching_type>
  match mode:  arg[1]=match arg[2]=<target_image> arg[3]=<csv_file> arg[4]=<N> arg[5]=<matching_type>

*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "dirent.h"
#include <opencv2/opencv.hpp>
#include "csv_util.h"
#include "features.hpp"

using namespace cv;
using namespace std;

// Helper structure for matching
struct Match
{
    float distance;
    int index;
    bool operator<(const Match& other) const
    {
        return distance<other.distance;
    }
};

// build features function
int buildFeatures(int argc, char* argv[])
{
    char dirname[256], csv_filename[256], buffer[256];
    DIR* dirp;
    struct dirent* dp;
    bool first_image = true;
    int image_count = 0;
    int matching_type = 1;

    // check for sufficient arguments
    if (argc<5)
    {
        printf("Usage: %s build <img_directory> <output_csv> <matching_type>\n", argv[0]);
        return -1;
    }

    strcpy_s(dirname, argv[2]);
    strcpy_s(csv_filename, argv[3]);
    matching_type = atoi(argv[4]);

    if (matching_type<1 || matching_type>6)
    {
        printf("Invalid matching type.\n");
        return -2;
    }

    switch (matching_type)
    {
    case 1: printf("Baseline Matching\n"); break;
    case 2: printf("Histogram Matching\n"); break;
    case 3: printf("Multi-histogram Matching\n"); break;
    case 4: printf("Texture and Color Matching\n"); break;
    case 5: printf("ResNet Features Matching\n"); break;
    case 6: printf("Custom Feature Matching\n"); break;
    }

    dirp = opendir(dirname);
    if (dirp == NULL)
    {
        printf("Cannot open directory %s\n", dirname);
        return -3;
    }

    while ((dp = readdir(dirp))!= NULL)
    {
        if (strstr(dp->d_name, ".jpg") || strstr(dp->d_name, ".png") ||
            strstr(dp->d_name, ".ppm") || strstr(dp->d_name, ".tif"))
        {
            strcpy_s(buffer, dirname);
            strcat_s(buffer, "/");
            strcat_s(buffer, dp->d_name);

            printf("Processing: %s\n", buffer);

            Mat image = imread(buffer);
            if (image.empty())
            {
                printf("Could not read image, skipping\n");
                continue;
            }

            vector<float> features;

            switch (matching_type)
            {
            case 1:
                features = extract_center(image);
                break;

            case 2:
                features = rg_histogram(image, 16);
                break;

            case 3:
            {
                int mid_row = image.rows/2;
                Mat top_half = image(Rect(0, 0, image.cols, mid_row));
                Mat bottom_half = image(Rect(0, mid_row, image.cols, image.rows-mid_row));

                vector<float> top_hist = rgb_histogram(top_half, 8);
                vector<float> bottom_hist = rgb_histogram(bottom_half, 8);

                features.reserve(top_hist.size()+bottom_hist.size());
                features.insert(features.end(), top_hist.begin(), top_hist.end());
                features.insert(features.end(), bottom_hist.begin(), bottom_hist.end());
                break;
            }

            case 4:
            {
                vector<float> color_hist = rgb_histogram(image, 8);
                vector<float> tex_hist = texture_histogram(image, 16);

                features.reserve(color_hist.size()+tex_hist.size());
                features.insert(features.end(), color_hist.begin(), color_hist.end());
                features.insert(features.end(), tex_hist.begin(), tex_hist.end());
                break;
            }

            case 6:
            {
                vector<float> color_hist = rgb_histogram(image, 8);
                vector<float> gabor_feat = gabor_features(image, 15, 32, 4);

                char dnn_csv[256];
                strcpy_s(dnn_csv, "olym_ResNet18.csv");

                vector<char*> dnn_filenames;
                vector<vector<float>> dnn_features;

                if (read_image_data_csv(dnn_csv, dnn_filenames, dnn_features, 0) == 0)
                {
                    int dnn_idx = -1;
                    for (int i=0; i<dnn_filenames.size(); i++)
                    {
                        if (strcmp(dnn_filenames[i], dp->d_name) == 0)
                        {
                            dnn_idx = i;
                            break;
                        }
                    }

                    if (dnn_idx!= -1)
                    {
                        features.reserve(color_hist.size()+gabor_feat.size()+dnn_features[dnn_idx].size());
                        features.insert(features.end(), color_hist.begin(), color_hist.end());
                        features.insert(features.end(), gabor_feat.begin(), gabor_feat.end());
                        features.insert(features.end(), dnn_features[dnn_idx].begin(), dnn_features[dnn_idx].end());
                    }

                    for (int i=0; i<dnn_filenames.size(); i++)
                    {
                        delete[] dnn_filenames[i];
                    }
                }
                break;
            }

            default:
                printf("Unknown matching type\n");
                break;
            }

            int reset = first_image ? 1 : 0;
            append_image_data_csv(csv_filename, dp->d_name, features, reset);
            first_image = false;
            image_count++;
        }
    }
    closedir(dirp);
    printf("\nProcessed %d images\n", image_count);
    return 0;
}

// Match images function
int matchImages(int argc, char* argv[])
{
    char target_filename[256], csv_filename[256];
    int N;
    int matching_type = 1;

    // check for sufficient arguments
    if (argc<6)
    {
        printf("Usage: %s match <target_image> <csv_file> <N> <matching_type>\n", argv[0]);
        return -1;
    }

    strcpy_s(target_filename, argv[2]);
    strcpy_s(csv_filename, argv[3]);
    N = atoi(argv[4]);
    matching_type = atoi(argv[5]);

    if (matching_type<1 || matching_type>6)
    {
        printf("Invalid matching type.\n");
        return -2;
    }

    Mat target = imread(target_filename);
    if (target.empty())
    {
        printf("Could not read target image %s\n", target_filename);
        return -1;
    }

    char img_dir[256];
    strcpy_s(img_dir, target_filename);
    char* last_slash = strrchr(img_dir, '/');
    if (last_slash == NULL)
    {
        last_slash = strrchr(img_dir, '\\');
    }
    if (last_slash!= NULL)
    {
        *(last_slash+1) = '\0';
    }
    else
    {
        strcpy_s(img_dir, "./");
    }

    vector<char*> filenames;
    vector<vector<float>> database_features;

    if (read_image_data_csv(csv_filename, filenames, database_features, 0)!= 0)
    {
        printf("Could not read feature database\n");
        return -1;
    }

    vector<float> target_features;

    switch (matching_type)
    {
    case 1:
        target_features = extract_center(target);
        break;

    case 2:
        target_features = rg_histogram(target, 16);
        break;

    case 3:
    {
        int mid_row = target.rows/2;
        Mat top_half = target(Rect(0, 0, target.cols, mid_row));
        Mat bottom_half = target(Rect(0, mid_row, target.cols, target.rows-mid_row));

        vector<float> top_hist = rgb_histogram(top_half, 8);
        vector<float> bottom_hist = rgb_histogram(bottom_half, 8);

        target_features.reserve(top_hist.size()+bottom_hist.size());
        target_features.insert(target_features.end(), top_hist.begin(), top_hist.end());
        target_features.insert(target_features.end(), bottom_hist.begin(), bottom_hist.end());
        break;
    }

    case 4:
    {
        vector<float> color_hist = rgb_histogram(target, 8);
        vector<float> tex_hist = texture_histogram(target, 16);

        target_features.reserve(color_hist.size()+tex_hist.size());
        target_features.insert(target_features.end(), color_hist.begin(), color_hist.end());
        target_features.insert(target_features.end(), tex_hist.begin(), tex_hist.end());
        break;
    }

    case 5:
    {
        char* target_name = strrchr(target_filename, '/');
        if (target_name == NULL)
        {
            target_name = strrchr(target_filename, '\\');
        }
        if (target_name == NULL)
        {
            target_name = target_filename;
        }
        else
        {
            target_name++;
        }

        int target_idx = -1;
        for (int i=0; i<filenames.size(); i++)
        {
            if (strcmp(filenames[i], target_name) == 0)
            {
                target_idx = i;
                break;
            }
        }

        if (target_idx == -1)
        {
            printf("Target image not found in ResNet\n");
            return -1;
        }

        target_features = database_features[target_idx];
        break;
    }

    case 6:
    {
        vector<float> color_hist = rgb_histogram(target, 8);
        vector<float> gabor_feat = gabor_features(target, 15, 32, 4);

        char* target_name = strrchr(target_filename, '/');
        if (target_name == NULL)
        {
            target_name = strrchr(target_filename, '\\');
        }
        if (target_name == NULL)
        {
            target_name = target_filename;
        }
        else
        {
            target_name++;
        }

        char dnn_csv[256];
        strcpy_s(dnn_csv, csv_filename);
        char* last_slash = strrchr(dnn_csv, '/');
        if (last_slash == NULL)
        {
            last_slash = strrchr(dnn_csv, '\\');
        }
        if (last_slash!= NULL)
        {
            *(last_slash+1) = '\0';
            strcat_s(dnn_csv, "olym_ResNet18.csv");
        }
        else
        {
            strcpy_s(dnn_csv, "olym_ResNet18.csv");
        }

        vector<char*> dnn_filenames;
        vector<vector<float>> dnn_db;

        if (read_image_data_csv(dnn_csv, dnn_filenames, dnn_db, 0) == 0)
        {
            int target_idx = -1;
            for (int i=0; i<dnn_filenames.size(); i++)
            {
                if (strcmp(dnn_filenames[i], target_name) == 0)
                {
                    target_idx = i;
                    break;
                }
            }

            if (target_idx!= -1)
            {
                target_features.reserve(color_hist.size()+gabor_feat.size()+dnn_db[target_idx].size());
                target_features.insert(target_features.end(), color_hist.begin(), color_hist.end());
                target_features.insert(target_features.end(), gabor_feat.begin(), gabor_feat.end());
                target_features.insert(target_features.end(), dnn_db[target_idx].begin(), dnn_db[target_idx].end());
            }

            for (int i=0; i<dnn_filenames.size(); i++)
            {
                delete[] dnn_filenames[i];
            }
        }
        break;
    }

    default:
        printf("Unknown matching type\n");
        break;
    }

    printf("Loaded %zu images from database\n\n", database_features.size());
    vector<Match> matches;

    for (int i=0; i<database_features.size(); i++)
    {
        float dist;
        if (matching_type == 1 || matching_type == 5)
        {
            dist = ssd(target_features, database_features[i]);
        }
        else if (matching_type == 3)
        {
            int half_size = target_features.size()/2;
            vector<float> target_top(target_features.begin(), target_features.begin()+half_size);
            vector<float> target_bottom(target_features.begin()+half_size, target_features.end());

            vector<float> db_top(database_features[i].begin(), database_features[i].begin()+half_size);
            vector<float> db_bottom(database_features[i].begin()+half_size, database_features[i].end());

            float dist_top = hist_inter(target_top, db_top);
            float dist_bottom = hist_inter(target_bottom, db_bottom);
            dist = (dist_top + dist_bottom)/2.0f;
        }
        else if (matching_type == 4)
        {
            int color_size = 8*8*8;
            vector<float> target_color(target_features.begin(), target_features.begin()+color_size);
            vector<float> target_texture(target_features.begin()+color_size, target_features.end());

            vector<float> db_color(database_features[i].begin(), database_features[i].begin()+color_size);
            vector<float> db_texture(database_features[i].begin()+color_size, database_features[i].end());

            float dist_color = hist_inter(target_color, db_color);
            float dist_texture = hist_inter(target_texture, db_texture);
            dist = (0.5f*dist_color) + (0.5f*dist_texture);
        }
        else if (matching_type == 6)
        {
            int color_size = 8*8*8;
            int gabor_size = 256;
            int dnn_size = 512;

            vector<float> target_color(target_features.begin(), target_features.begin()+color_size);
            vector<float> target_gabor(target_features.begin()+color_size,
                                       target_features.begin()+color_size+gabor_size);
            vector<float> target_dnn(target_features.begin()+gabor_size+color_size, target_features.end());

            vector<float> db_color(database_features[i].begin(), database_features[i].begin()+color_size);
            vector<float> db_gabor(database_features[i].begin()+color_size,
                                   database_features[i].begin()+color_size+gabor_size);
            vector<float> db_dnn(database_features[i].begin()+gabor_size+color_size, database_features[i].end());

            float dist_color = hist_inter(target_color, db_color);
            float dist_gabor = ssd(target_gabor, db_gabor);
            float dist_dnn = ssd(target_dnn, db_dnn);
            dist = (0.1f*dist_color) + (0.15f*dist_gabor/gabor_size) + (0.75f*dist_dnn/dnn_size);
        }
        else
        {
            dist = hist_inter(target_features, database_features[i]);
        }

        Match m;
        m.distance = dist;
        m.index = i;
        matches.push_back(m);
    }

    sort(matches.begin(), matches.end());

    namedWindow("Target Image", WINDOW_AUTOSIZE);
    cv::imshow("Target Image", target);

    int displayed = 0;
    for (int i=0; i<matches.size() && displayed<N; i++)
    {
        int idx = matches[i].index;
        char* db_name = filenames[idx];

        char* target_name = strrchr(target_filename, '/');
        if (target_name == NULL)
        {
            target_name = strrchr(target_filename, '\\');
        }
        if (target_name == NULL)
        {
            target_name = target_filename;
        }
        else
        {
            target_name++;
        }

        if (strcmp(db_name, target_name) == 0)
        {
            continue;
        }

        displayed++;
        printf("%2d. %s (Distance: %.4f)\n", displayed, db_name, matches[i].distance);

        char match_path[256];
        strcpy_s(match_path, img_dir);
        strcat_s(match_path, db_name);

        Mat match_img = imread(match_path);
        if (!match_img.empty())
        {
            char window_name[256];
            sprintf_s(window_name, "Match %d (dist: %.4f)", displayed, matches[i].distance);
            namedWindow(window_name, WINDOW_AUTOSIZE);
            cv::imshow(window_name, match_img);
        }
    }

    cv::waitKey(0);
    cv::destroyAllWindows();

    for (int i=0; i<filenames.size(); i++)
    {
        delete[] filenames[i];
    }

    return 0;
}

int main(int argc, char* argv[])
{
    if (argc<2)
    {
        printf("Usage:\n");
        printf("build: %s build <img_directory> <output_csv> <matching_type>\n", argv[0]);
        printf("match: %s match <target_image> <csv_file> <N> <matching_type>\n", argv[0]);
        return -1;
    }

    if (strcmp(argv[1], "build") == 0)
    {
        return buildFeatures(argc, argv);
    }
    else if (strcmp(argv[1], "match") == 0)
    {
        return matchImages(argc, argv);
    }
    else
    {
        printf("Invalid mode'\n");
        return -1;
    }
}