
__kernel void SSD_depth_estimator (__global unsigned char* leftImage,
                                   __global unsigned char* rightImage,
                                   __global unsigned char* leftRightImage,
                                   __global unsigned char* leftRightCopy,
                                   __global unsigned char* rightLeftImage,
                                   __local uchar4* localLeft,
                                   __local uchar4* localRight,
                                   int width,
                                   int height,
                                   int win_size,
                                   int MAX_DISP)
{
	size_t col = get_global_id (0);
	size_t row = get_global_id (1);

	size_t lcol = get_local_id (0) + win_size;      //offset for local memory loading
	size_t lrow = get_local_id (1) + win_size;

	size_t lwidth = get_local_size (0);

	int lDifference, rDifference;                   //difference between 2 pixel values
	int lSum, rSum, sum;                            //sum of differences
	int L_MIN_SUM, R_MIN_SUM;                       //minimum sums found for SSD block matching
	int l_best_d, r_best_d;                         //best disparity values for images
	int k, l, leftIndex, rightIndex, d;

    if(col >= win_size && col <= width-win_size && row >= win_size && row <= height-win_size)
    {
        //loads image data to local memory
        //amount loaded is == (workgroup width + window_size*2 + MAX_DISP)*(workgroup height + window_size*2)
        //note that most of the pixels are loaded to local memory multiple times. This does not break anything but results in useless loading
        for(k = -win_size; k <= win_size; k++)
        {
            for(l = -win_size; l <= win_size+MAX_DISP; l++)
            {
                leftIndex = lcol + l + (lrow + k)*(lwidth+2*win_size+MAX_DISP);
                localLeft[leftIndex].x = leftImage [col + l + (row + k)*width];
            }
            for(l = -win_size-MAX_DISP; l <= win_size; l++)
            {
                if(col >= MAX_DISP+win_size) {
                    rightIndex = lcol + MAX_DISP + l + (lrow + k)*(lwidth+2*win_size+MAX_DISP)+1;
                    localRight[rightIndex].x = rightImage [col + l + (row + k)*width];
                }
            }
        }
        L_MIN_SUM = 10000000;
        R_MIN_SUM = 10000000;
        l_best_d = 0;
        r_best_d = 0;
        //goes trough all the disparity values
        for(d = 0; d < MAX_DISP; d++) {
            lSum = 0;
            rSum = 0;
            for(k = -win_size; k <= win_size; k++)
            {
                for(l = -win_size; l <= win_size; l++)
                {
                    //does not calculate if not enough pixels to left of pixel
                    if(col >= win_size+MAX_DISP)
                    {
                        leftIndex = lcol + l + (lrow + k)*(lwidth+2*win_size+MAX_DISP);
                        rightIndex = lcol + MAX_DISP - d + l + (lrow + k)*(lwidth+2*win_size+MAX_DISP);

                        //global memory method
                        //lDifference = leftImage [col + l + (row + k)*width] - rightImage [col - d + l + (row + k)*width];

                        lDifference = localLeft [leftIndex].x - localRight [rightIndex].x;
                        lSum += lDifference*lDifference;
                    }
                    //does not calculate if not enough pixels to right of pixel
                    if(col <= win_size-MAX_DISP)
                    {
                        leftIndex = lcol + d + l + (lrow + k)*(lwidth+2*win_size+MAX_DISP);
                        rightIndex = lcol + MAX_DISP + l + (lrow + k)*(lwidth+2*win_size+MAX_DISP);

                        //global memory method
                        //rDifference = leftImage [col + d + l + (row + k)*width] - rightImage [col + l + (row + k)*width];

                        rDifference = localLeft [leftIndex].x - localRight [rightIndex].x;
                        rSum += rDifference*rDifference;
                    }
                }
            }
            //update minimum sum
            if(L_MIN_SUM > lSum) {
                L_MIN_SUM = lSum;
                l_best_d = d;
            }
            if(R_MIN_SUM > rSum) {
                R_MIN_SUM = rSum;
                r_best_d = d;
            }
        }
        //otherwise set pixel to 0 (when not enough pixels to the left/right of image point)
        if(L_MIN_SUM == 0)
        {
            l_best_d = 0;
        }
        if(R_MIN_SUM == 0)
        {
            r_best_d = 0;
        }
        // save the image to 2 places for occlusion (so the occlusion filled pixels do not affect the averaging)
        leftRightImage [col + row*width] = l_best_d*255/MAX_DISP;
        leftRightCopy [col + row*width] = l_best_d*255/MAX_DISP;
        rightLeftImage [col + row*width] = r_best_d*255/MAX_DISP;

        // cross check for inconsistencies
        if(col >= win_size+MAX_DISP) {
            //Pixel_left[index] - Pixel_right[index-Pixel_left[index]]]
            if(abs_diff((int)leftImage [col + row*width], (int)rightImage [col + row*width - leftRightImage [col + row*width]]) > 125)
            {
                leftRightImage [col + row*width] = 0;
                leftRightCopy [col + row*width] = 0;
            }
        }
    }
    else
    {
        leftRightImage [col + row*width] = 0;
    }
    //wait for all of the work items to finish writing to the result matrices
    barrier(CLK_GLOBAL_MEM_FENCE);

    int occlusion_window = 5;
    // occlusion filler with a 11x11 window averaging
    if(col >= win_size+MAX_DISP && col <= width-win_size && row >= win_size && row <= height-win_size)
    {
        if( leftRightImage [col + row*width] == 0 )
        {
           sum = 0;
            for(k = -occlusion_window; k <= occlusion_window; k++)
            {
                for(l = -occlusion_window; l <= occlusion_window; l++)
                {
                    sum += leftRightCopy [col + l + (row + k)*width];
               }
            }
            //take the average of window pixels expect the center since it is always 0
            sum = sum/((occlusion_window*2+1)*(occlusion_window*2+1)-1);
            leftRightImage [col + row*width] = sum;
        }
    }

}

