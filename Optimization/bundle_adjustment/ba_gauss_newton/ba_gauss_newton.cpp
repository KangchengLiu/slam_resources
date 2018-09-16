//
// Created by xiang on 12/21/17.
//

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Eigen;

#include <vector>
#include <fstream>
#include <iostream>
#include <iomanip>

#include <sophus/se3.h>

using namespace std;

typedef vector<Vector3d, Eigen::aligned_allocator<Vector3d>> VecVector3d;
typedef vector<Vector2d, Eigen::aligned_allocator<Vector3d>> VecVector2d;
typedef Matrix<double, 6, 1> Vector6d;

string p3d_file = "../p3d.txt";
string p2d_file = "../p2d.txt";

int main(int argc, char **argv) {

    VecVector2d p2d;
    VecVector3d p3d;
    Matrix3d K;
    double fx = 520.9, fy = 521.0, cx = 325.1, cy = 249.7;
    K << fx, 0, cx, 0, fy, cy, 0, 0, 1;

    // load points in to p3d and p2d
    // START YOUR CODE HERE

    ifstream fileP2D(p2d_file);
    ifstream fileP3D(p3d_file);
    if(!fileP2D.is_open() || !fileP3D.is_open()){
        cout << "fileP2D or fileP3D open FAILED!" << endl;
        return -1;
    }
    double u=0, v=0;
    double x=0, y=0, z=0;
    while(!fileP2D.eof()){
        if(fileP2D.fail())
            break;
        fileP2D >> u >> v;
        fileP3D >> x >> y >> z;
        p2d.push_back(Vector2d(u,v));
        p3d.push_back(Vector3d(x,y,z));
    }

    // END YOUR CODE HERE
    assert(p3d.size() == p2d.size());

    int iterations = 100;
    double cost = 0, lastCost = 0;
    int nPoints = p3d.size();
    cout << "points: " << nPoints << endl;

    Sophus::SE3 T_esti; // estimated pose

    for (int iter = 0; iter < iterations; iter++) {

        Matrix<double, 6, 6> H = Matrix<double, 6, 6>::Zero();
        Vector6d b = Vector6d::Zero();

        cost = 0;
        // compute cost
        for (int i = 0; i < nPoints; i++) {
            // compute cost for p3d[I] and p2d[I]
            // START YOUR CODE HERE

            Vector2d uv = p2d[i];
            Vector3d Pc = T_esti * p3d[i];

            double x = Pc[0];
            double y = Pc[1];
            double z = Pc[2];

            double x2 = x*x;
            double y2 = y*y;
            double z2 = z*z;

            Vector2d e = uv - (K * Pc).hnormalized();

            cost += e.squaredNorm();

	        // END YOUR CODE HERE

	        // compute jacobian
            Matrix<double, 2, 6> J;
            // START YOUR CODE HERE

            J(0,0) = -fx/z;
            J(0,1) =  0;
            J(0,2) =  fx*x/z2;
            J(0,3) =  fx*x*y/z2;
            J(0,4) = -fx-fx*x2/z2;
            J(0,5) =  fx*y/z;
            J(1,0) =  0;
            J(1,1) = -fy/z;
            J(1,2) =  fy*y/z2;
            J(1,3) =  fy+fy*y2/z2;
            J(1,4) = -fy*x*y/z2;
            J(1,5) = -fy*x/z;

	        // END YOUR CODE HERE

            H +=  J.transpose() * J;
            b += -J.transpose() * e;
        }

	    // solve dx
        Vector6d dx;
        // START YOUR CODE HERE
        dx = H.ldlt().solve(b);
        // END YOUR CODE HERE

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }

        // update your estimation
        // START YOUR CODE HERE

        T_esti *= Sophus::SE3::exp(dx);

        // END YOUR CODE HERE

        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cout.precision(12) << cost << endl;
    }

    cout << "estimated pose: \n" << T_esti.matrix() << endl;
    return 0;
}
