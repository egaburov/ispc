#include <stdio.h>
#define blockIndex0 (blockIdx.x*4 + (threadIdx.x >> 5))
#define blockIndex1 (blockIdx.y)
#define vectorWidth (32)
#define vectorIndex (threadIdx.x & 31)

  int __device__ __forceinline__
mandel(float c_re, float c_im, int count) 
{
  float z_re = c_re, z_im = c_im;
  int i;
  for (i = 0; i < count; ++i) {
    if (z_re * z_re + z_im * z_im > 4.0f)
      break;

    float new_re = z_re*z_re - z_im*z_im;
    float new_im = 2.0f * z_re * z_im;
    {
      z_re = c_re + new_re;
      z_im = c_im + new_im;
    }
  }

  return i;
}

extern "C"
__global__ void mandelbrot_scanline(
    float x0, float dx, 
    float y0, float dy,
    int width, int height, 
    int xspan, int yspan,
    int maxIterations, int output[]) 
{
  const int xstart = blockIndex0 * xspan;
  const int xend   = min(xstart  + xspan, width);

  const int ystart = blockIndex1 * yspan;
  const int yend   = min(ystart  + yspan, height);

  for (int yi = ystart; yi < yend; yi++)
    for (int xi = xstart; xi < xend; xi += vectorWidth)
    {
      const float x = x0 + (xi + vectorIndex) * dx;
      const float y = y0 +  yi              * dy;

      const int res = mandel(x,y,maxIterations);
      const int index = yi * width + (xi + vectorIndex);
      if (xi + vectorIndex < xend)
        output[index] = res;
    }
}

