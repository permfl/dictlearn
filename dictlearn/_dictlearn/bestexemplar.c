#include <stdlib.h>
#include <stdio.h>
#include "bestexemplar.h"


void bestexemplar(double* image, size_t height, size_t width,
                  const double* patch, unsigned int patch_height,
                  unsigned int patch_width, unsigned int* to_fill,
                  unsigned int* source, unsigned int* best)
{
    size_t last_row, last_col; /* Number of patches per row and col*/
    unsigned int row, col; /* Current row and col in global image*/
    size_t curr_pixel; /* Index to current pixel in global image*/
    size_t p_row, p_col; /* Row and col index in global image for patch with
                            curr_pixel as upper left corner */
    size_t patch_px;
    double best_error; /* Value of best error so far*/
    double patch_err; /* Error for current patch */
    double dist; /* Error between pixel in image and corr pixel in patch*/

    last_row = height - patch_height + 1;
    last_col = width - patch_width + 1;
    best_error = 1000000.0;

    for(row = 0; row < last_row; row++) {
        for(col = 0; col < last_col; col++) {

            patch_px = 0;
            for(p_row = row; p_row < row + patch_height; p_row++) {
                for(p_col = col; p_col < col + patch_width; p_col++) {
                    curr_pixel = p_row*width + p_col;
                    if(source[curr_pixel] == 0)
                        goto skipPatch;

                    if(to_fill[patch_px] == 0) {
                        dist = image[curr_pixel] - patch[patch_px];
                        patch_err = patch_err + dist*dist;
                    }

                    patch_px++;
                }
            }

            if(patch_err < best_error) {

                best_error = patch_err;
                best[0] = row;
                best[1] = row + patch_height;
                best[2] = col;
                best[3] = col + patch_width;
            }

            skipPatch:
                patch_err = 0.0;
        }
    }
}


void bestexemplar_3d(double* image, size_t height, size_t width, size_t depth,
                     const double* patch, unsigned int patch_height, 
                     unsigned int patch_width, unsigned int patch_depth,
                     unsigned int* to_fill, unsigned int* source, 
                     unsigned int* best)
{
    /* Number of patches per row and col and depth*/
    size_t last_row, last_col, last_depth; 
    
    /* Current row and col in global image*/
    unsigned int row, col, dep; 
    
    /* Index to current pixel in global image*/
    size_t curr_pixel; 
    
    /* 
        Row and col index in global image for patch with
        curr_pixel as upper left corner 
    */
    size_t p_row, p_col, p_dep; 
    size_t patch_px;
    double best_error; /* Value of best error so far*/
    double patch_err; /* Error for current patch */
    double dist; /* Error between pixel in image and corr pixel in patch*/

    int count;

    last_row = height - patch_height + 1;
    last_col = width - patch_width + 1;
    last_depth = depth - patch_depth + 1;
    best_error = 100000000.0;   

    count = 0;
    for(row = 0; row < last_row; row++) {
        for(col = 0; col < last_col; col++) {
            for(dep = 0; dep < last_depth; dep++) {
                patch_px = 0;
                /*if(count < 1000) {
                    printf("%d  -  %d\n", count, row*width + col*depth + dep);
                }
                else {
                    return;
                }

                count++;*/

                for(p_row = row; p_row < row + patch_height; p_row++) {
                    for(p_col = col; p_col < col + patch_width; p_col++) {
                        for(p_dep = dep; p_dep < dep + patch_depth; p_dep++) {
                            curr_pixel = p_row*width + p_col*depth + p_dep;

                            
                            if(source[curr_pixel] == 0)
                                goto skipPatch; 


                            if(to_fill[patch_px] == 0) {
                                dist = image[curr_pixel] - patch[patch_px];
                                patch_err = patch_err + dist*dist;
                            }

                            patch_px++;
                        }
                    }
                }

                /*//printf("%f\n", patch_err);*/
                if(patch_err < best_error) {
                    best_error = patch_err;
                    best[0] = row;
                    best[1] = row + patch_height;
                    best[2] = col;
                    best[3] = col + patch_width;
                    best[4] = dep;
                    best[5] = dep + patch_depth;
                }

                skipPatch:
                    patch_err = 0.0;
            }
        }
    }
}