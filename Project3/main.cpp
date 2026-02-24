/*
Arpit Gandhi
February 2026

Main file for real time 2D object recognition system using hand-crafted (HC) features and CNN embeddings.
'n' key to add HC feature entry to database
'm' key to add entry to CNN embedding entry to database
'r' key to toggle region visualization
't' key to toggle threshold image visualization
'e' key to create confusion matrices for HC and CNN
'q' key to quit
*/

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>
#include "utilities.h"

using namespace std;
using namespace cv;

// Custom Thresholding ISODATA (k=2)
int isodataThreshold(const Mat& gray)
{
	// sampling 1 in every 16 pixels
    vector<float> samples;
    samples.reserve((gray.rows/4)*(gray.cols/4));
    for (int r = 0; r < gray.rows; r+= 4)
    {
        const uchar* rptr = gray.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; c+= 4)
        {
            samples.push_back(rptr[c]);
        }
    }

    // means start at extremes and converge toward the two histogram peaks
    float m1 = 0.f, m2 = 255.f;
    for (int iter = 0; iter < 100; ++iter)
    {
        float sum1 = 0, sum2 = 0;
        int cnt1 = 0, cnt2 = 0;
        for (float v : samples)
        {
            if (abs(v-m1) <= abs(v-m2))
            {
                sum1+= v;
                ++cnt1;
            }
            else
            {
                sum2+= v;
                ++cnt2;
            }
        }
        float nm1 = cnt1 ? sum1/cnt1 : m1;
        float nm2 = cnt2 ? sum2/cnt2 : m2;
        if (abs(nm1-m1) < 0.5f && abs(nm2-m2) < 0.5f)
        {
            break;
        }
        m1 = nm1;
        m2 = nm2;
    }
    return (int)((m1+m2)/2.f);
}

// 255 = foreground, 0 = background
Mat customThreshold(const Mat& gray)
{
    int thresh = isodataThreshold(gray);
    Mat binary(gray.size(), CV_8UC1);
    for (int r = 0; r < gray.rows; ++r)
    {
        const uchar* sptr = gray.ptr<uchar>(r);
        uchar* dptr = binary.ptr<uchar>(r);
        for (int c = 0; c < gray.cols; ++c)
        {
            dptr[c] = (sptr[c] < thresh) ? 255 : 0;
        }
    }
    return binary;
}

// Feature Extraction
static const int FEAT_DIM = 9;

struct RegionFeatures
{
    double percentFilled, aspectRatio;
    double hu[7];
    double cx, cy, theta, axisLen;
    RotatedRect obb;
    float minE1, maxE1, minE2, maxE2;
};

RegionFeatures computeFeatures(const Mat& mask8u)
{
    RegionFeatures f{};
    Moments m = moments(mask8u, true);
    if (m.m00 < 1.0)
    {
        return f;
    }

    f.cx = m.m10/m.m00;
    f.cy = m.m01/m.m00;

    double mu20 = m.mu20/m.m00, mu02 = m.mu02/m.m00, mu11 = m.mu11/m.m00;
    f.theta = 0.5*atan2(2.0*mu11, mu20-mu02);

    double raw[7];
    HuMoments(m, raw);
    for (int i = 0; i < 7; ++i)
    {
        f.hu[i] = (raw[i]!= 0.0) ? copysign(log10(abs(raw[i])), raw[i]) : 0.0;
    }

    vector<Point> pts;
    findNonZero(mask8u, pts);
    if (pts.empty())
    {
        return f;
    }
    f.obb = minAreaRect(pts);

    float w = f.obb.size.width, h = f.obb.size.height;
    float major = max(w, h), minor = min(w, h);
    f.percentFilled = (major*minor > 0) ? (m.m00/(major*minor)) : 0.0;
    f.aspectRatio = (minor > 0) ? (double(major)/minor) : 1.0;
    f.axisLen = major/2.0;

    // compute axis projections needed by prepEmbeddingImage
    double cosT = cos(f.theta), sinT = sin(f.theta);
    f.minE1 = 1e9f; f.maxE1 = -1e9f; f.minE2 = 1e9f; f.maxE2 = -1e9f;
    for (auto& p : pts)
    {
        float dx = (float)(p.x-f.cx), dy = (float)(p.y-f.cy);
        float e1 = (float)(dx*cosT + dy*sinT);
        float e2 = (float)(-dx*sinT + dy*cosT);
        if (e1 < f.minE1)
        {
            f.minE1 = e1;
        }
        if (e1 > f.maxE1)
        {
            f.maxE1 = e1;
        }
        if (e2 < f.minE2)
        {
            f.minE2 = e2;
        }
        if (e2 > f.maxE2) 
        {
            f.maxE2 = e2;
        }
    }
    return f;
}

vector<double> featToVec(const RegionFeatures& f)
{
    return {f.percentFilled, f.aspectRatio,
            f.hu[0], f.hu[1], f.hu[2], f.hu[3], f.hu[4], f.hu[5], f.hu[6]};
}

// Object Database (CSV)
struct DBEntry {
    string label;
    vector<double> vec;
};

void saveEntry(const string& path, const string& lbl, const vector<double>& v)
{
    ofstream ofs(path, ios::app);
    ofs << lbl;
    for (double x : v)
    {
        ofs << "," << x;
    }
    ofs << "\n";
}

vector<DBEntry> loadDB(const string& path)
{
    vector<DBEntry> db;
    ifstream ifs(path);
    if (!ifs)
    {
        return db;
    }
    string line;
    while (getline(ifs, line))
    {
        if (line.empty())
        {
            continue;
        }
        istringstream ss(line);
        DBEntry e;
        getline(ss, e.label, ',');
        string tok;
        while (getline(ss, tok, ','))
        {
            e.vec.push_back(stod(tok));
        }
        if ((int)e.vec.size() == FEAT_DIM)
        {
            db.push_back(e);
        }
    }
    return db;
}

// Scaled Euclidean Nearest-Neighbour Classifier
vector<double> computeStdDevs(const vector<DBEntry>& db)
{
    vector<double> stds(FEAT_DIM, 1.0);
    if (db.size() < 2)
    {
        return stds;
    }
    for (int i = 0; i < FEAT_DIM; ++i)
    {
        double mean = 0;
        for (auto& e : db)
        {
            mean+= e.vec[i];
        }
        mean/= db.size();
        double var = 0;
        for (auto& e : db)
        {
            var+= (e.vec[i]-mean)*(e.vec[i]-mean);
        }
        var/= db.size();
        stds[i] = (var > 1e-10) ? sqrt(var) : 1.0;
    }
    return stds;
}

pair<string, double> classify(const vector<double>& query, const vector<DBEntry>& db, const vector<double>& stds)
{
    double best = numeric_limits<double>::max();
    string bestLbl = "unknown";
    for (auto& e : db)
    {
        double d = 0;
        for (int i = 0; i < FEAT_DIM; ++i)
        {
            double diff = (query[i]-e.vec[i])/stds[i];
            d+= diff*diff;
        }
        if (d < best)
        {
            best = d;
            bestLbl = e.label;
        }
    }
    return {bestLbl, best};
}

// CNN Embedding Database (CSV)
static const int CNN_DIM = 512;

struct CNNEntry {
    string label;
    vector<float> vec;
};

void saveCNNEntry(const string& path, const string& lbl, const Mat& emb)
{
    ofstream ofs(path, ios::app);
    ofs << lbl;
    const float* ptr = emb.ptr<float>(0);
    for (int i = 0; i < CNN_DIM; ++i)
    {
        ofs << "," << ptr[i];
    }
    ofs << "\n";
}

vector<CNNEntry> loadCNNDB(const string& path)
{
    vector<CNNEntry> db;
    ifstream ifs(path);
    if (!ifs)
    {
        return db;
    }
    string line;
    while (getline(ifs, line))
    {
        if (line.empty())
        {
            continue;
        }
        istringstream ss(line);
        CNNEntry e;
        getline(ss, e.label, ',');
        string tok;
        while (getline(ss, tok, ','))
        {
            e.vec.push_back(stof(tok));
        }
        if ((int)e.vec.size() == CNN_DIM)
        {
            db.push_back(e);
        }
    }
    return db;
}

// Sum-squared difference distance for CNN embeddings
pair<string, double> classifyCNN(const Mat& emb, const vector<CNNEntry>& db)
{
    double best = numeric_limits<double>::max();
    string bestLbl = "unknown";
    const float* qp = emb.ptr<float>(0);
    for (auto& e : db)
    {
        double d = 0;
        for (int i = 0; i < CNN_DIM; ++i)
        {
            double diff = qp[i]-e.vec[i];
            d+= diff*diff;
        }
        if (d < best)
        {
            best = d; 
            bestLbl = e.label;
        }
    }
    return {bestLbl, best};
}

// Confusion Matrix
struct ConfusionMatrix
{
    vector<string> labels;
    vector<vector<int>> matrix;

    int indexOf(const string& lbl)
    {
        for (int i = 0; i < (int)labels.size(); ++i)
        {
            if (labels[i] == lbl)
            {
                return i;
            }
        }
        labels.push_back(lbl);
        for (auto& row : matrix)
        {
            row.push_back(0);
        }
        matrix.push_back(vector<int>(labels.size(), 0));
        return (int)labels.size()-1;
    }

    void record(const string& trueLabel, const string& predLabel)
    {
        matrix[indexOf(trueLabel)][indexOf(predLabel)]++;
    }

    void print(const string& title, const string& savePath = "") const
    {
        if (labels.empty())
        {
            cout << title << ": no data.\n";
            return;
        }

        int cw = 6;
        for (auto& l : labels)
        {
            cw = max(cw, (int)l.size()+2);
        }

        auto printTo = [&](ostream& os)
            {
                os << "\n" << title << "\n" << string(cw, ' ');
                for (auto& l : labels)
                {
                    os << setw(cw) << l;
                }
                os << "   <- Predicted\n" << string(cw*((int)labels.size()+1)+14, '-') << "\n";

                for (int ti = 0; ti < (int)labels.size(); ++ti)
                {
                    os << setw(cw) << labels[ti];
                    for (int pi = 0; pi < (int)labels.size(); ++pi)
                    {
                        string cell = to_string(matrix[ti][pi]);
                        if (ti == pi && matrix[ti][pi] > 0)
                        {
                            cell += "*";
                        }
                        os << setw(cw) << cell;
                    }
                    os << "\n";
                }
                os << "Rows = True label,  * = correct\n\nPer-class accuracy:\n";

                for (int ti = 0; ti < (int)labels.size(); ++ti)
                {
                    int total = 0;
                    for (int v : matrix[ti])
                    {
                        total+= v;
                    }
                    double acc = total > 0 ? 100.0*matrix[ti][ti]/total : 0.0;
                    os << "  " << setw(cw) << labels[ti] << " : " << matrix[ti][ti]
                        << "/" << total << "  (" << fixed << setprecision(1) << acc << "%)\n";
                }
            };

        printTo(cout);
        if (!savePath.empty())
        {
            ofstream ofs(savePath);
            if (ofs)
            {
                printTo(ofs);
                cout << "Saved to " << savePath << "\n";
            }
        }
    }
};

//  Drawing helpers
static const Scalar PALETTE[8] = {
    {60,60,255},{60,255,60},{255,60,60},{0,220,220},
    {220,0,220},{220,220,0},{180,100,40},{100,40,180}
};

void drawOBB(Mat& img, const RotatedRect& obb, Scalar col)
{
    Point2f pts[4];
    obb.points(pts);
    for (int i = 0; i < 4; ++i)
    {
        line(img, pts[i], pts[(i+1)%4], col, 2, LINE_AA);
    }
}

void drawAxis(Mat& img, double cx, double cy, double theta, double len, Scalar col)
{
    double dx = cos(theta)*len, dy = sin(theta)*len;
    arrowedLine(img, Point(cvRound(cx-dx), cvRound(cy-dy)),
        Point(cvRound(cx+dx), cvRound(cy+dy)), col, 2, LINE_AA, 0, 0.12);
}

Mat coloriseRegions(const Mat& labels, const vector<int>& ids, int H, int W)
{
    Mat vis = Mat::zeros(H, W, CV_8UC3);
    for (int r = 0; r < H; ++r)
    {
        const int* labelPtr = labels.ptr<int>(r);
        Vec3b* visPtr = vis.ptr<Vec3b>(r);
        for (int c = 0; c < W; ++c)
        {
            int lbl = labelPtr[c];
            for (int ri = 0; ri < (int)ids.size(); ++ri)
            {
                if (lbl == ids[ri])
                {
                    Scalar col = PALETTE[ri%8];
                    visPtr[c] = {(uchar)col[0],(uchar)col[1],(uchar)col[2]};
                    break;
                }
            }
        }
    }
    return vis;
}


int main()
{
    const string DB_PATH = "object_db.csv";
    const string CNN_DB_PATH = "cnn_db.csv";
    const string NET_PATH = "resnet18-v2-7.onnx";

    const int MIN_AREA = 3000;
    const int MAX_REGIONS = 5;

    VideoCapture cap(0);
    if (!cap.isOpened())
    {
        cerr << "Cannot open webcam.\n";
        return 1;
    }
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);

    auto db = loadDB(DB_PATH);
    auto stds = computeStdDevs(db);
    auto cnnDB = loadCNNDB(CNN_DB_PATH);

    dnn::Net net;
    bool netLoaded = false;
    try
    {
        net = dnn::readNet(NET_PATH);
        netLoaded = true;
        cout << "ResNet18 loaded.\n";
    }
    catch (...)
    {
        cerr << "could not load ResNet18 " << NET_PATH << ".\n";
    }

    bool showThresh = true, showRegions = true, hasRegion = false;
    ConfusionMatrix cmHC, cmCNN;
    string lastHCpred, lastCNNpred;

    Mat kernel = getStructuringElement(MORPH_ELLIPSE, {11,11});

    while (true)
    {
        Mat frame, gray, blurred, binary, cleaned;
        cap >> frame;
        if (frame.empty())
        {
            break;
        }
        const int W = frame.cols, H = frame.rows;

        cvtColor(frame, gray, COLOR_BGR2GRAY);
        GaussianBlur(gray, blurred, {5,5}, 1.5);
        binary = customThreshold(blurred);
        Mat kernelOpen = getStructuringElement(MORPH_ELLIPSE, {3,3});
        morphologyEx(binary, cleaned, MORPH_CLOSE, kernel);
        morphologyEx(cleaned, cleaned, MORPH_OPEN, kernelOpen);

        Mat labels, stats, centroids;
        int nLabels = connectedComponentsWithStats(cleaned, labels, stats, centroids);

        struct RegInfo {
            int id;
            int area;
        };

        vector<RegInfo> valid;
        for (int i = 1; i < nLabels; ++i)
        {
            const int* sp = stats.ptr<int>(i);
            int area = sp[CC_STAT_AREA], x = sp[CC_STAT_LEFT], y = sp[CC_STAT_TOP];
            int rw = sp[CC_STAT_WIDTH], rh = sp[CC_STAT_HEIGHT];
            if (area < MIN_AREA)
            {
                continue;
            }
            if (x <= 1 || y <= 1 || x+rw >= W-1 || y+rh >= H-1)
            {
                continue;
            }
            valid.push_back({i,area});
        }
        sort(valid.begin(), valid.end(), [](auto& a, auto& b) { return a.area > b.area; });
        if ((int)valid.size() > MAX_REGIONS)
        {
            valid.resize(MAX_REGIONS);
        }

        vector<int> validIds;
        for (auto& v : valid)
        {
            validIds.push_back(v.id);
        }

        Mat outputViz = frame.clone();
        Mat regionViz;
        if (showRegions)
        {
            regionViz = coloriseRegions(labels, validIds, H, W);
        }

        hasRegion = false;
        lastHCpred = ""; lastCNNpred = "";

        for (int ri = 0; ri < (int)valid.size(); ++ri)
        {
            int id = valid[ri].id;
            Mat mask = (labels == id);
            mask.convertTo(mask, CV_8UC1, 255);

            RegionFeatures feat = computeFeatures(mask);
            if (feat.axisLen < 1.0)
            {
                continue;
            }

            drawOBB(outputViz, feat.obb, PALETTE[ri % 8]);
            drawAxis(outputViz, feat.cx, feat.cy, feat.theta, feat.axisLen, {0,230,230});

            string hcLbl = "no db";
            if (!db.empty())
            {
                auto res = classify(featToVec(feat), db, stds);
                hcLbl = classify(featToVec(feat), db, stds).first;
            }

            string cnnLbl = "cnn off";
            if (netLoaded && !cnnDB.empty())
            {
                Mat embimage, embedding;
                prepEmbeddingImage(frame, embimage, (int)feat.cx, (int)feat.cy, (float)feat.theta,
                    feat.minE1, feat.maxE1, feat.minE2, feat.maxE2, 0);
                if (!embimage.empty())
                {
                    getEmbedding(embimage, embedding, net, 0);
                    cnnLbl = classifyCNN(embedding, cnnDB).first;
                }
            }

            if (ri == 0)
            {
                lastHCpred = hcLbl; lastCNNpred = cnnLbl; hasRegion = true;
            }

            int tx = cvRound(feat.cx)-50;
            int ty = max(cvRound(feat.cy-feat.axisLen)-40, 20);
            putText(outputViz, "HC:  " + hcLbl, {tx,ty}, FONT_HERSHEY_SIMPLEX, 0.75, {0,255,80}, 2, LINE_AA);
            putText(outputViz, "CNN: " + cnnLbl, {tx,ty+28}, FONT_HERSHEY_SIMPLEX, 0.75, {80,200,255}, 2, LINE_AA);

            char buf[64];
            snprintf(buf, sizeof(buf), "fill=%.2f AR=%.2f", feat.percentFilled, feat.aspectRatio);
            putText(outputViz, buf, {tx,ty+52}, FONT_HERSHEY_SIMPLEX, 0.5, {200,200,0}, 1, LINE_AA);
        }

        if (hasRegion)
        {
            putText(outputViz, "[e] to record test", {10,30}, FONT_HERSHEY_SIMPLEX, 0.6, {0,255,255}, 2);
        }

        int totalTests = 0;
        for (auto& r : cmHC.matrix)
        {
            for (int v : r)
            {
                totalTests+= v;
            }
        }
        string status = "HC:" + to_string(db.size()) + " CNN:" + to_string(cnnDB.size()) +
            " Tests:" + to_string(totalTests) +
            "  [e]record [n]train [m]CNN-train [q]quit";
        putText(outputViz, status, {10,H-10}, FONT_HERSHEY_SIMPLEX, 0.42, {180,180,180}, 1);

        imshow("Object Recognition", outputViz);
        if (showThresh)
        {
            imshow("Threshold Image", binary);
			imshow("Cleaned Image", cleaned);
        }
        if (showRegions && !regionViz.empty())
        {
            imshow("Region Map", regionViz);
        }

        int key = waitKey(1) & 0xFF;
        if (key == 'q')
        {
            break;
        }
        if (key == 't')
        {
            showThresh = !showThresh;
            if (!showThresh)
            {
                destroyWindow("Threshold Image");
                destroyWindow("Cleaned Image");
            }
            else
            {
                imshow("Threshold Image", binary);
                imshow("Cleaned Image", cleaned);
            }
        }
        if (key == 'r')
        {
            showRegions = !showRegions;
            if (!showRegions)
            {
                destroyWindow("Region Map");
            }
        }

        if (key == 'e')
        {
            if (!hasRegion)
            {
                cout << "[Eval] No region visible.\n";
            }
            else
            {
                cout << "[Eval] HC: " << lastHCpred << "  CNN: " << lastCNNpred << "\n";
                cout << "[Eval] Enter TRUE label: ";
                string trueLabel;
                getline(cin, trueLabel);
                if (!trueLabel.empty())
                {
                    cmHC.record(trueLabel, lastHCpred);
                    cmCNN.record(trueLabel, lastCNNpred);
                    cout << "[Eval] Recorded. Total: " << totalTests+1 << "\n";
                }
            }
        }

        if (key == 'n')
        {
            if (valid.empty())
            {
                cout << "[Train-HC] No region.\n";
                continue;
            }
            Mat mask = (labels == valid[0].id);
            mask.convertTo(mask, CV_8UC1, 255);
            vector<double> fvec = featToVec(computeFeatures(mask));
            cout << "Feature vector:\n";
            cout << "  fill=" << fvec[0] << " AR=" << fvec[1] << "\n";
            for (int i = 0; i < 7; ++i)
                cout << "hu["<<i<<"]=" << fvec[2+i] << "\n";
            cout << "[Train-HC] Enter label: ";
            string lbl;
            getline(cin, lbl);
            if (lbl.empty())
            {
                continue;
            }
            saveEntry(DB_PATH, lbl, fvec);
            db = loadDB(DB_PATH);
            stds = computeStdDevs(db);
            cout << "[Train-HC] Saved '" << lbl << "'. DB=" << db.size() << "\n";
        }

        if (key == 'm' && netLoaded)
        {
            if (valid.empty())
            {
                cout << "[Train-CNN] No region.\n";
                continue;
            }
            Mat mask = (labels == valid[0].id);
            mask.convertTo(mask, CV_8UC1, 255);
            RegionFeatures feat = computeFeatures(mask);
            Mat embimage, embedding;
            prepEmbeddingImage(frame, embimage,
                (int)feat.cx, (int)feat.cy, (float)feat.theta,
                feat.minE1, feat.maxE1, feat.minE2, feat.maxE2, 0);
            if (embimage.empty())
            {
                cout << "[Train-CNN] ROI too small.\n";
                continue;
            }
            getEmbedding(embimage, embedding, net, 0);
            cout << "[Train-CNN] Enter label: ";
            string lbl;
            getline(cin, lbl);
            if (lbl.empty())
            {
                continue;
            }
            saveCNNEntry(CNN_DB_PATH, lbl, embedding);
            cnnDB = loadCNNDB(CNN_DB_PATH);
            cout << "[Train-CNN] Saved '" << lbl << "'. CNN DB=" << cnnDB.size() << "\n";
        }
    }

    cmHC.print("Hand-Crafted Confusion Matrix", "confusion_hc.txt");
    cmCNN.print("CNN Confusion Matrix", "confusion_cnn.txt");

    cv::destroyAllWindows();
    return 0;
}