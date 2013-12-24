/*
  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/


#include "deferred.h"
#include <stdio.h>
#include <assert.h>

#define programCount 32
#define programIndex (threadIdx.x & 31)
#define taskIndex (blockIdx.x*4 + (threadIdx.x >> 5))
#define taskCount (gridDim.x*4)
#define warpIdx (threadIdx.x >> 5)

#define int32 int
#define int16 short
#define int8 char

__device__ static inline float clamp(float v, float low, float high) 
{
      return min(max(v, low), high);
}

struct InputDataArrays
{
    float *zBuffer;
    unsigned int16 *normalEncoded_x; // half float
    unsigned int16 *normalEncoded_y; // half float
    unsigned int16 *specularAmount; // half float
    unsigned int16 *specularPower; // half float
    unsigned int8 *albedo_x; // unorm8
    unsigned int8 *albedo_y; // unorm8
    unsigned int8 *albedo_z; // unorm8
    float *lightPositionView_x;
    float *lightPositionView_y;
    float *lightPositionView_z;
    float *lightAttenuationBegin;
    float *lightColor_x;
    float *lightColor_y;
    float *lightColor_z;
    float *lightAttenuationEnd;
};

struct InputHeader
{
    float cameraProj[4][4];
    float cameraNear;
    float cameraFar;

    int32 framebufferWidth;
    int32 framebufferHeight;
    int32 numLights;
    int32 inputDataChunkSize;
    int32 inputDataArrayOffsets[idaNum];
};


///////////////////////////////////////////////////////////////////////////
// Common utility routines

__device__
static inline float
dot3(float x, float y, float z, float a, float b, float c) {
    return (x*a + y*b + z*c);
}


#if 0
template<typename T, int N>
struct Uniform
{
  static __shared__ T shdata[128];
  T data[(N-1)/programCount+1];

  __device__ inline const T get(const int i) const
  {
    const int  elemIdx = i & (programCount-1);
    const int chunkIdx = i >> 5;
    return __shfl(data[chunkIdx], elemIdx);
  }
  
  __device__ inline void set(const int i, const T value) const
  {
    const int  elemIdx = i & (programCount-1);
    const int chunkIdx = i >> 5;
    shdata[elemIdx] = value;
    data[chunkIdx]  = shdata[programIndex];
  }
}
#endif


__device__
static inline void
normalize3(float x, float y, float z, float &ox, float &oy, float &oz) {
    float n = rsqrt(x*x + y*y + z*z);
    ox = x * n;
    oy = y * n;
    oz = z * n;
}

__device__ inline
static float reduce_min(float value)
{
#pragma unroll
  for (int i = 4; i >=0; i--)
    value = min(value, __shfl_xor(value, 1<<i, 32));
  return value;
}
__device__ inline
static float reduce_max(float value)
{
#pragma unroll
  for (int i = 4; i >=0; i--)
    value = max(value, __shfl_xor(value, 1<<i, 32));
  return value;
}
__device__ inline
static int reduce_sum(int value)
{
#pragma unroll
  for (int i = 4; i >=0; i--)
    value +=  __shfl_xor(value, 1<<i, 32);
  return value;
}
static __device__ __forceinline__ uint shfl_scan_add_step(uint partial, uint up_offset)
{
  uint result;
  asm(
      "{.reg .u32 r0;"
      ".reg .pred p;"
      "shfl.up.b32 r0|p, %1, %2, 0;"
      "@p add.u32 r0, r0, %3;"
      "mov.u32 %0, r0;}"
      : "=r"(result) : "r"(partial), "r"(up_offset), "r"(partial));
  return result;
}
static __device__ __forceinline__ int inclusive_scan_warp(const int value)
{
  uint sum = value;
#pragma unroll
  for(int i = 0; i < 5; ++i)
    sum = shfl_scan_add_step(sum, 1 << i);
  return sum - value;
}


static __device__ __forceinline__ int lanemask_lt()
{
  int mask;
  asm("mov.u32 %0, %lanemask_lt;" : "=r" (mask));
  return mask;
}
static __device__ __forceinline__ int2 warpBinExclusiveScan(const bool p)
{
  const unsigned int b = __ballot(p);
  return make_int2(__popc(b & lanemask_lt()), __popc(b));
}





__device__
static inline float
Unorm8ToFloat32(unsigned int8 u) {
    return (float)u * (1.0f / 255.0f);
}


__device__
static inline unsigned int8
Float32ToUnorm8(float f) {
    return (unsigned int8)(f * 255.0f);
}


__device__
static inline void
ComputeZBounds(
     int32 tileStartX,  int32 tileEndX,
     int32 tileStartY,  int32 tileEndY,
    // G-buffer data
     float zBuffer[],
     int32 gBufferWidth,
    // Camera data
     float cameraProj_33,  float cameraProj_43,
     float cameraNear,  float cameraFar,
    // Output
     float &minZ,
     float &maxZ
    )
{
    // Find Z bounds
    float laneMinZ = cameraFar;
    float laneMaxZ = cameraNear;
    for ( int32 y = tileStartY; y < tileEndY; ++y) {
        for ( int xb = tileStartX; xb < tileEndX; xb += programCount)
        {
          const int x = xb + programIndex;
          if (x >= tileEndX) break;
            // Unproject depth buffer Z value into view space
            float z = zBuffer[y * gBufferWidth + x];
            float viewSpaceZ = cameraProj_43 / (z - cameraProj_33);

            // Work out Z bounds for our samples
            // Avoid considering skybox/background or otherwise invalid pixels
            if ((viewSpaceZ < cameraFar) && (viewSpaceZ >= cameraNear)) {
                laneMinZ = min(laneMinZ, viewSpaceZ);
                laneMaxZ = max(laneMaxZ, viewSpaceZ);
            }
        }
    }
    minZ = reduce_min(laneMinZ);
    maxZ = reduce_max(laneMaxZ);
}


__device__
static inline  int32
IntersectLightsWithTileMinMax(
     int32 tileStartX,  int32 tileEndX,
     int32 tileStartY,  int32 tileEndY,
    // Tile data
     float minZ,
     float maxZ,
    // G-buffer data
     int32 gBufferWidth,  int32 gBufferHeight,
    // Camera data
     float cameraProj_11,  float cameraProj_22,
    // Light Data
     int32 numLights,
     float light_positionView_x_array[],
     float light_positionView_y_array[],
     float light_positionView_z_array[],
     float light_attenuationEnd_array[],
    // Output
     int32 tileLightIndices[]
    )
{
     float gBufferScale_x = 0.5f * (float)gBufferWidth;
     float gBufferScale_y = 0.5f * (float)gBufferHeight;
        
     float frustumPlanes_xy[4] = {
        -(cameraProj_11 * gBufferScale_x),
         (cameraProj_11 * gBufferScale_x),
         (cameraProj_22 * gBufferScale_y),
        -(cameraProj_22 * gBufferScale_y) };
     float frustumPlanes_z[4] = {
         tileEndX - gBufferScale_x,
        -tileStartX + gBufferScale_x,
         tileEndY - gBufferScale_y,
        -tileStartY + gBufferScale_y };

    for ( int i = 0; i < 4; ++i) {
         float norm = rsqrt(frustumPlanes_xy[i] * frustumPlanes_xy[i] + 
                                   frustumPlanes_z[i] * frustumPlanes_z[i]);
        frustumPlanes_xy[i] *= norm;
        frustumPlanes_z[i] *= norm;
    }

     int32 tileNumLights = 0;

    for ( int lightIndexB = 0; lightIndexB < numLights; lightIndexB += programCount)
    {
      const int lightIndex = lightIndexB + programIndex;

        float light_positionView_z = light_positionView_z_array[lightIndex];
        float light_attenuationEnd = light_attenuationEnd_array[lightIndex];
        float light_attenuationEndNeg = -light_attenuationEnd;

        float d = light_positionView_z - minZ;
        bool inFrustum = (d >= light_attenuationEndNeg);

        d = maxZ - light_positionView_z;
        inFrustum = inFrustum && (d >= light_attenuationEndNeg);
        
        // This seems better than cif(!inFrustum) ccontinue; here since we
        // don't actually need to mask the rest of this function - this is
        // just a greedy early-out.  Could also structure all of this as
        // nested if() statements, but this a bit easier to read
        int active = 0;
        if ((inFrustum)) {
            float light_positionView_x = light_positionView_x_array[lightIndex];
            float light_positionView_y = light_positionView_y_array[lightIndex];

            d = light_positionView_z * frustumPlanes_z[0] + 
                light_positionView_x * frustumPlanes_xy[0];
            inFrustum = inFrustum && (d >= light_attenuationEndNeg);

            d = light_positionView_z * frustumPlanes_z[1] + 
                light_positionView_x * frustumPlanes_xy[1];
            inFrustum = inFrustum && (d >= light_attenuationEndNeg);

            d = light_positionView_z * frustumPlanes_z[2] + 
                light_positionView_y * frustumPlanes_xy[2];
            inFrustum = inFrustum && (d >= light_attenuationEndNeg);

            d = light_positionView_z * frustumPlanes_z[3] + 
                light_positionView_y * frustumPlanes_xy[3];
            inFrustum = inFrustum && (d >= light_attenuationEndNeg);
        
            // Pack and store intersecting lights
#if 0
            if (inFrustum) {
                tileNumLights += packed_store_active(&tileLightIndices[tileNumLights], 
                                                     lightIndex);
            }
#else
            if (inFrustum)
            {
              active = 1;
            }
#endif
        }
#if 1
        if (lightIndex >= numLights) 
          active = 0;

#if 0
        const int idx = tileNumLights + inclusive_scan_warp(active);
        const int nactive = reduce_sum(active);
#else
        const int2 res = warpBinExclusiveScan(active);
        const int idx = tileNumLights + res.x;
        const int nactive = res.y;
#endif
        if (active)
          tileLightIndices[idx] = lightIndex;
        tileNumLights += nactive;
#endif
    }

    return tileNumLights;
}


__device__
static inline   int32
IntersectLightsWithTile(
     int32 tileStartX,  int32 tileEndX,
     int32 tileStartY,  int32 tileEndY,
     int32 gBufferWidth,  int32 gBufferHeight,
    // G-buffer data
     float zBuffer[],
    // Camera data
     float cameraProj_11,  float cameraProj_22,
     float cameraProj_33,  float cameraProj_43,
     float cameraNear,  float cameraFar,
    // Light Data
     int32 numLights,
     float light_positionView_x_array[],
     float light_positionView_y_array[],
     float light_positionView_z_array[],
     float light_attenuationEnd_array[],
    // Output
     int32 tileLightIndices[]
    )
{
     float minZ, maxZ;
    ComputeZBounds(tileStartX, tileEndX, tileStartY, tileEndY,
        zBuffer, gBufferWidth, cameraProj_33, cameraProj_43, cameraNear, cameraFar,
        minZ, maxZ);


     int32 tileNumLights = IntersectLightsWithTileMinMax(
        tileStartX, tileEndX, tileStartY, tileEndY, minZ, maxZ,
        gBufferWidth, gBufferHeight, cameraProj_11, cameraProj_22,
        MAX_LIGHTS, light_positionView_x_array, light_positionView_y_array, 
        light_positionView_z_array, light_attenuationEnd_array,
        tileLightIndices);

    return tileNumLights;
}


__device__
static inline void
ShadeTile(
     int32 tileStartX,  int32 tileEndX,
     int32 tileStartY,  int32 tileEndY,
     int32 gBufferWidth,  int32 gBufferHeight,
    const  InputDataArrays &inputData,
    // Camera data
     float cameraProj_11,  float cameraProj_22,
     float cameraProj_33,  float cameraProj_43,
    // Light list
     int32 tileLightIndices[],
     int32 tileNumLights,
    // UI
     bool visualizeLightCount,
    // Output
     unsigned int8 framebuffer_r[],
     unsigned int8 framebuffer_g[],
     unsigned int8 framebuffer_b[]
    )
{
    if (tileNumLights == 0 || visualizeLightCount) {
         unsigned int8 c = (unsigned int8)(min(tileNumLights << 2, 255));
        for ( int32 y = tileStartY; y < tileEndY; ++y) {
            for ( int xb = tileStartX ; xb < tileEndX; xb += programCount)
            { 
              const int x = xb + programIndex;
              if (x >= tileEndX) continue;
                int32 framebufferIndex = (y * gBufferWidth + x);
                framebuffer_r[framebufferIndex] = c;
                framebuffer_g[framebufferIndex] = c;
                framebuffer_b[framebufferIndex] = c;
            }
        }
    } else {
         float twoOverGBufferWidth = 2.0f / gBufferWidth;
         float twoOverGBufferHeight = 2.0f / gBufferHeight;
        
        for ( int32 y = tileStartY; y < tileEndY; ++y) {
             float positionScreen_y = -(((0.5f + y) * twoOverGBufferHeight) - 1.f);

            for ( int xb = tileStartX ; xb < tileEndX; xb += programCount)
            { 
              const int x = xb + programIndex;
//              if (x >= tileEndX) break;
                int32 gBufferOffset = y * gBufferWidth + x;
                
                // Reconstruct position and (negative) view vector from G-buffer
                float surface_positionView_x, surface_positionView_y, surface_positionView_z;
                float Vneg_x, Vneg_y, Vneg_z;

                float z = inputData.zBuffer[gBufferOffset];

                // Compute screen/clip-space position
                // NOTE: Mind DX11 viewport transform and pixel center!
                float positionScreen_x = (0.5f + (float)(x)) * 
                    twoOverGBufferWidth - 1.0f;

                // Unproject depth buffer Z value into view space
                surface_positionView_z = cameraProj_43 / (z - cameraProj_33);
                surface_positionView_x = positionScreen_x * surface_positionView_z / 
                    cameraProj_11;
                surface_positionView_y = positionScreen_y * surface_positionView_z / 
                    cameraProj_22;
                
                // We actually end up with a vector pointing *at* the
                // surface (i.e. the negative view vector)
                normalize3(surface_positionView_x, surface_positionView_y, 
                           surface_positionView_z, Vneg_x, Vneg_y, Vneg_z);

                // Reconstruct normal from G-buffer
                float surface_normal_x, surface_normal_y, surface_normal_z;
                float normal_x = __half2float(inputData.normalEncoded_x[gBufferOffset]);
                float normal_y = __half2float(inputData.normalEncoded_y[gBufferOffset]);
                    
                float f = (normal_x - normal_x * normal_x) + (normal_y - normal_y * normal_y);
                float m = sqrt(4.0f * f - 1.0f);
                    
                surface_normal_x = m * (4.0f * normal_x - 2.0f);
                surface_normal_y = m * (4.0f * normal_y - 2.0f);
                surface_normal_z = 3.0f - 8.0f * f;

                // Load other G-buffer parameters
                float surface_specularAmount = 
                    __half2float(inputData.specularAmount[gBufferOffset]);
                float surface_specularPower  = 
                    __half2float(inputData.specularPower[gBufferOffset]);
                float surface_albedo_x = Unorm8ToFloat32(inputData.albedo_x[gBufferOffset]);
                float surface_albedo_y = Unorm8ToFloat32(inputData.albedo_y[gBufferOffset]);
                float surface_albedo_z = Unorm8ToFloat32(inputData.albedo_z[gBufferOffset]);
                
                float lit_x = 0.0f;
                float lit_y = 0.0f;
                float lit_z = 0.0f;
                for ( int32 tileLightIndex = 0; tileLightIndex < tileNumLights; 
                     ++tileLightIndex) {
                     int32 lightIndex = tileLightIndices[tileLightIndex];
                                        
                    // Gather light data relevant to initial culling
                     float light_positionView_x = 
                        inputData.lightPositionView_x[lightIndex];
                     float light_positionView_y = 
                        inputData.lightPositionView_y[lightIndex];
                     float light_positionView_z = 
                        inputData.lightPositionView_z[lightIndex];
                     float light_attenuationEnd = 
                        inputData.lightAttenuationEnd[lightIndex];
                    
                    // Compute light vector
                    float L_x = light_positionView_x - surface_positionView_x;
                    float L_y = light_positionView_y - surface_positionView_y;
                    float L_z = light_positionView_z - surface_positionView_z;

                    float distanceToLight2 = dot3(L_x, L_y, L_z, L_x, L_y, L_z);
                    
                    // Clip at end of attenuation
                    float light_attenutaionEnd2 = light_attenuationEnd * light_attenuationEnd;

                    if (distanceToLight2 < light_attenutaionEnd2) {                    
                        float distanceToLight = sqrt(distanceToLight2);

                        // HLSL "rcp" is allowed to be fairly inaccurate
                        float distanceToLightRcp = 1.0f/distanceToLight;
                        L_x *= distanceToLightRcp;
                        L_y *= distanceToLightRcp;
                        L_z *= distanceToLightRcp;

                        // Start computing brdf
                        float NdotL = dot3(surface_normal_x, surface_normal_y, 
                                           surface_normal_z, L_x, L_y, L_z);
                    
                        // Clip back facing
                        if (NdotL > 0.0f) {
                             float light_attenuationBegin = 
                                inputData.lightAttenuationBegin[lightIndex];

                            // Light distance attenuation (linstep)
                            float lightRange = (light_attenuationEnd - light_attenuationBegin);
                            float falloffPosition = (light_attenuationEnd - distanceToLight);
                            float attenuation = min(falloffPosition / lightRange, 1.0f);

                            float H_x = (L_x - Vneg_x);
                            float H_y = (L_y - Vneg_y);
                            float H_z = (L_z - Vneg_z);
                            normalize3(H_x, H_y, H_z, H_x, H_y, H_z);
                    
                            float NdotH = dot3(surface_normal_x, surface_normal_y, 
                                               surface_normal_z, H_x, H_y, H_z);
                            NdotH = max(NdotH, 0.0f);

                            float specular = pow(NdotH, surface_specularPower);
                            float specularNorm = (surface_specularPower + 2.0f) * 
                                (1.0f / 8.0f);
                            float specularContrib = surface_specularAmount * 
                                specularNorm * specular;

                            float k = attenuation * NdotL * (1.0f + specularContrib);
                    
                             float light_color_x = inputData.lightColor_x[lightIndex];
                             float light_color_y = inputData.lightColor_y[lightIndex];
                             float light_color_z = inputData.lightColor_z[lightIndex];

                            float lightContrib_x = surface_albedo_x * light_color_x;
                            float lightContrib_y = surface_albedo_y * light_color_y;
                            float lightContrib_z = surface_albedo_z * light_color_z;

                            lit_x += lightContrib_x * k;
                            lit_y += lightContrib_y * k;
                            lit_z += lightContrib_z * k;
                        }
                    }
                }

                // Gamma correct
                // These pows are pretty slow right now, but we can do
                // something faster if really necessary to squeeze every
                // last bit of performance out of it
                float gamma = 1.0 / 2.2f;
                lit_x = pow(clamp(lit_x, 0.0f, 1.0f), gamma);
                lit_y = pow(clamp(lit_y, 0.0f, 1.0f), gamma);
                lit_z = pow(clamp(lit_z, 0.0f, 1.0f), gamma);
                
                framebuffer_r[gBufferOffset] = Float32ToUnorm8(lit_x);
                framebuffer_g[gBufferOffset] = Float32ToUnorm8(lit_y);
                framebuffer_b[gBufferOffset] = Float32ToUnorm8(lit_z);
            }
        }
    }
}


///////////////////////////////////////////////////////////////////////////
// Static decomposition

extern "C" __global__ void
RenderTile( int num_groups_x,  int num_groups_y,
           const  InputHeader *inputHeaderPtr,
           const  InputDataArrays *inputDataPtr,
            int visualizeLightCount,
           // Output
            unsigned int8 framebuffer_r[],
            unsigned int8 framebuffer_g[],
            unsigned int8 framebuffer_b[]) {
  if (taskIndex >= taskCount) return;

  const  InputHeader inputHeader = *inputHeaderPtr;
  const  InputDataArrays inputData = *inputDataPtr;
     int32 group_y = taskIndex / num_groups_x;
     int32 group_x = taskIndex % num_groups_x;

     int32 tile_start_x = group_x * MIN_TILE_WIDTH;
     int32 tile_start_y = group_y * MIN_TILE_HEIGHT;
     int32 tile_end_x = tile_start_x + MIN_TILE_WIDTH;
     int32 tile_end_y = tile_start_y + MIN_TILE_HEIGHT;

     int framebufferWidth = inputHeader.framebufferWidth;
     int framebufferHeight = inputHeader.framebufferHeight;
     float cameraProj_00 = inputHeader.cameraProj[0][0];
     float cameraProj_11 = inputHeader.cameraProj[1][1];
     float cameraProj_22 = inputHeader.cameraProj[2][2];
     float cameraProj_32 = inputHeader.cameraProj[3][2];

    // Light intersection: figure out which lights illuminate this tile.
#if 0
     int tileLightIndices[MAX_LIGHTS];  // Light list for the tile
#else
     __shared__ int tileLightIndicesFull[4*MAX_LIGHTS];  // Light list for the tile
     int *tileLightIndices = &tileLightIndicesFull[warpIdx*MAX_LIGHTS];
#endif
     int numTileLights = 
        IntersectLightsWithTile(tile_start_x, tile_end_x, 
                                tile_start_y, tile_end_y,
                                framebufferWidth, framebufferHeight,
                                inputData.zBuffer,
                                cameraProj_00, cameraProj_11,
                                cameraProj_22, cameraProj_32,
                                inputHeader.cameraNear, inputHeader.cameraFar,
                                MAX_LIGHTS,
                                inputData.lightPositionView_x, 
                                inputData.lightPositionView_y, 
                                inputData.lightPositionView_z, 
                                inputData.lightAttenuationEnd,
                                tileLightIndices);

    // And now shade the tile, using the lights in tileLightIndices
    ShadeTile(tile_start_x, tile_end_x, tile_start_y, tile_end_y,
              framebufferWidth, framebufferHeight, inputData,
              cameraProj_00, cameraProj_11, cameraProj_22, cameraProj_32,
              tileLightIndices, numTileLights, visualizeLightCount, 
              framebuffer_r, framebuffer_g, framebuffer_b);
}


