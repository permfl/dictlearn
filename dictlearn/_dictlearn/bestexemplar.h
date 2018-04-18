void bestexemplar(double* image, size_t height, size_t width,
                  const double* patch, unsigned int patch_height,
                  unsigned int patch_width, unsigned int* to_fill,
                  unsigned int* source, unsigned int* best);


void bestexemplar_3d(double* image, size_t height, size_t width, size_t depth,
                     const double* patch, unsigned int patch_height, 
                     unsigned int patch_width, unsigned int patch_depth,
                     unsigned int* to_fill, unsigned int* source, 
                     unsigned int* best);