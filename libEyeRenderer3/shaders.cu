//
// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//  * Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//  * Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//  * Neither the name of NVIDIA CORPORATION nor the names of its
//    contributors may be used to endorse or promote products derived
//    from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
// OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// This codebase has been edited by Blayze Millward for the CompoundRay
// system and does not reflect the original codebase provided by NVIDIA
// corporation.
//
#include <optix.h>

#include <cuda/LocalGeometry.h>
#include <cuda/random.h>
#include <sutil/vec_math.h>
#include "GlobalParameters.h"

#include <stdint.h>

#include <stdio.h>

// For each camera Datatype:
#include "cameras/PerspectiveCameraDataTypes.h"
#include "cameras/PanoramicCameraDataTypes.h"
#include "cameras/GenericCameraDataTypes.h"
#include "cameras/OrthographicCameraDataTypes.h"
#include "cameras/CompoundEyeDataTypes.h"

// cuRand
#include <curand_kernel.h>

__constant__ float FWHM_SD_RATIO = 2.35482004503094938202313865291f;//939927549477137877164107704505151300005317709396985361683627673754162213494315716402473805711790020883378678441772367335067327119300654086099581027060701147250592490674309776452246690206347679431657862550790224141333488894447689644236226579600412626548283966926341892712473657396439184227529340027195703289818425375703612253952994171698822696215836693931109079884506177990740279369004153115665698570697083992256

extern "C"
{
__constant__ globalParameters::LaunchParams params;
}


//------------------------------------------------------------------------------
//
// GGX/smith shading helpers
//
//------------------------------------------------------------------------------

__device__ float3 schlick( const float3 spec_color, const float V_dot_H )
{
    return spec_color + ( make_float3( 1.0f ) - spec_color ) * powf( 1.0f - V_dot_H, 5.0f );
}


__device__ float vis( const float N_dot_L, const float N_dot_V, const float alpha )
{
    const float alpha_sq = alpha*alpha;

    const float ggx0 = N_dot_L * sqrtf( N_dot_V*N_dot_V * ( 1.0f - alpha_sq ) + alpha_sq );
    const float ggx1 = N_dot_V * sqrtf( N_dot_L*N_dot_L * ( 1.0f - alpha_sq ) + alpha_sq );

    return 2.0f * N_dot_L * N_dot_V / (ggx0+ggx1);
}


__device__ float ggxNormal( const float N_dot_H, const float alpha )
{
    const float alpha_sq   = alpha*alpha;
    const float N_dot_H_sq = N_dot_H*N_dot_H;
    const float x          = N_dot_H_sq*( alpha_sq - 1.0f ) + 1.0f;
    return alpha_sq/( M_PIf*x*x );
}


__device__ float3 linearize( float3 c )
{
    return make_float3(
            powf( c.x, 2.2f ),
            powf( c.y, 2.2f ),
            powf( c.z, 2.2f )
            );
}


//------------------------------------------------------------------------------
//
//
//
//------------------------------------------------------------------------------


static __forceinline__ __device__ void traceRadiance(
        OptixTraversableHandle      handle,
        float3                      ray_origin,
        float3                      ray_direction,
        float                       tmin,
        float                       tmax,
        globalParameters::PayloadRadiance*   payload
        )
{
    uint32_t u0=0, u1=0, u2=0, u3=0;
    optixTrace(
            handle,
            ray_origin, ray_direction,
            tmin,
            tmax,
            0.0f,                     // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_NONE,
            globalParameters::RAY_TYPE_RADIANCE,        // SBT offset
            globalParameters::RAY_TYPE_COUNT,           // SBT stride
            globalParameters::RAY_TYPE_RADIANCE,        // missSBTIndex
            u0, u1, u2, u3 );

     payload->result.x = __int_as_float( u0 );
     payload->result.y = __int_as_float( u1 );
     payload->result.z = __int_as_float( u2 );
     payload->depth    = u3;
}


static __forceinline__ __device__ bool traceOcclusion(
        OptixTraversableHandle handle,
        float3                 ray_origin,
        float3                 ray_direction,
        float                  tmin,
        float                  tmax
        )
{
    uint32_t occluded = 0u;
    optixTrace(
            handle,
            ray_origin,
            ray_direction,
            tmin,
            tmax,
            0.0f,                    // rayTime
            OptixVisibilityMask( 1 ),
            OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            globalParameters::RAY_TYPE_OCCLUSION,      // SBT offset
            globalParameters::RAY_TYPE_COUNT,          // SBT stride
            globalParameters::RAY_TYPE_OCCLUSION,      // missSBTIndex
            occluded );
    return occluded;
}


__forceinline__ __device__ void setPayloadResult( float3 p )
{
    optixSetPayload_0( __float_as_int( p.x ) );
    optixSetPayload_1( __float_as_int( p.y ) );
    optixSetPayload_2( __float_as_int( p.z ) );
}


__forceinline__ __device__ void setPayloadOcclusion( bool occluded )
{
    optixSetPayload_0( static_cast<uint32_t>( occluded ) );
}


__forceinline__ __device__ uchar4 make_color( const float3&  c )
{
    const float gamma = 2.2f;
    return make_uchar4(
            static_cast<uint8_t>( powf( clamp( c.x, 0.0f, 1.0f ), 1.0/gamma )*255.0f ),
            static_cast<uint8_t>( powf( clamp( c.y, 0.0f, 1.0f ), 1.0/gamma )*255.0f ),
            static_cast<uint8_t>( powf( clamp( c.z, 0.0f, 1.0f ), 1.0/gamma )*255.0f ),
            255u
            );
}


//------------------------------------------------------------------------------
//
//  Ray Generation Programs
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__pinhole()
{
    PerspectiveCameraPosedData* posedData = (PerspectiveCameraPosedData*)optixGetSbtDataPointer();
    const uint3  launch_idx      = optixGetLaunchIndex();
    const uint3  launch_dims     = optixGetLaunchDimensions();

    //
    // Generate camera ray
    //
    const float2 subpixel_jitter = make_float2(0.0f);// No subpixel jitter here.

    const float2 d = 2.0f * make_float2(
            ( static_cast<float>( launch_idx.x ) + subpixel_jitter.x ) / static_cast<float>( launch_dims.x ),
            ( static_cast<float>( launch_idx.y ) + subpixel_jitter.y ) / static_cast<float>( launch_dims.y )
            ) - 1.0f;

    const LocalSpace& ls = posedData->localSpace;
    const float3 scale = posedData->specializedData.scale;
    const float3 ray_direction = ls.zAxis*scale.z + d.x*ls.xAxis*scale.x + d.y*ls.yAxis*scale.y;
    const float3 ray_origin    = posedData->position;

    //
    // Trace camera ray
    //
    globalParameters::PayloadRadiance payload;
    payload.result = make_float3( 0.0f );
    payload.importance = 1.0f;
    payload.depth = 0.0f;

    traceRadiance(
            params.handle,
            ray_origin,
            ray_direction,
            0.01f,  // tmin       // TODO: smarter offset
            1e16f,  // tmax
            &payload );

    //
    // Update results
    //
    const uint32_t image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
    params.frame_buffer[image_index] = make_color(payload.result);
}

extern "C" __global__ void __raygen__panoramic()
{
    PanoramicCameraPosedData* posedData = (PanoramicCameraPosedData*)optixGetSbtDataPointer();
    const uint3  launch_idx      = optixGetLaunchIndex();
    const uint3  launch_dims     = optixGetLaunchDimensions();

    //
    // Generate camera ray
    //
    const float2 subpixel_jitter = make_float2(0.0f);// No subpixel jitter here

    const float2 d = 2.0f * make_float2(
            ( static_cast<float>( launch_idx.x ) + subpixel_jitter.x ) / static_cast<float>( launch_dims.x ),
            ( static_cast<float>( launch_idx.y ) + subpixel_jitter.y ) / static_cast<float>( launch_dims.y )
            ) - 1.0f;

    const float2 angles = d * make_float2(-M_PIf, M_PIf/2.0f) + make_float2(M_PIf/2.0f, 0.0f);
    const float cosY = cos(angles.y);
    const float3 originalDir = make_float3(cos(angles.x)*cosY, sin(angles.y), sin(angles.x)*cosY);
    const float3 lxAxis = posedData->localSpace.xAxis;
    const float3 lyAxis = posedData->localSpace.yAxis;
    const float3 lzAxis = posedData->localSpace.zAxis;
    const float3 ray_direction = normalize(originalDir.x * lxAxis + originalDir.y * lyAxis + originalDir.z * lzAxis);
    //const float3 ray_direction = normalize(posedData->localSpace.transform(originalDir));
    const float3 ray_origin    = posedData->position + ray_direction*posedData->specializedData.startRadius;

    //
    // Trace camera ray
    //
    globalParameters::PayloadRadiance payload;
    payload.result = make_float3( 0.0f );
    payload.importance = 1.0f;
    payload.depth = 0.0f;

    traceRadiance(
            params.handle,
            ray_origin,
            ray_direction,
            0.01f,  // tmin       // TODO: smarter offset
            1e16f,  // tmax
            &payload );

    //
    // Update results
    //
    const uint32_t image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
    params.frame_buffer[image_index] = make_color(payload.result);
}

extern "C" __global__ void __raygen__orthographic()
{
    OrthographicCameraPosedData* posedData = (OrthographicCameraPosedData*)optixGetSbtDataPointer();
    const uint3  launch_idx      = optixGetLaunchIndex();
    const uint3  launch_dims     = optixGetLaunchDimensions();

    //
    // Generate camera ray
    //
    const float2 subpixel_jitter = make_float2(0.0f);// No subpixel jitter here.

    const float2 d = 2.0f * make_float2(
            ( static_cast<float>( launch_idx.x ) + subpixel_jitter.x ) / static_cast<float>( launch_dims.x ),
            ( static_cast<float>( launch_idx.y ) + subpixel_jitter.y ) / static_cast<float>( launch_dims.y )
            ) - 1.0f;

    const LocalSpace& ls = posedData->localSpace;
    const float2 scale = posedData->specializedData.scale;
    const float3 ray_direction = ls.zAxis;
    const float3 ray_origin    = posedData->position + d.x*ls.xAxis*scale.x + d.y*ls.yAxis*scale.y;

    //
    // Trace camera ray
    //
    globalParameters::PayloadRadiance payload;
    payload.result = make_float3( 0.0f );
    payload.importance = 1.0f;
    payload.depth = 0.0f;

    traceRadiance(
            params.handle,
            ray_origin,
            ray_direction,
            0.01f,  // tmin       // TODO: smarter offset
            1e16f,  // tmax
            &payload );

    //
    // Update results
    //
    const uint32_t image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
    params.frame_buffer[image_index] = make_color(payload.result);
}

//------------------------------------------------------------------------------
//
//  Ommatidial Ray Projection Generation Programs
//
//------------------------------------------------------------------------------

__device__ float3 getSummedOmmatidiumData(const uint32_t ommatidiumIndex, CompoundEyeData& eyeData)
{
  float3 summation = make_float3(0.0f);
  for(int i = 0; i<eyeData.samplesPerOmmatidium; i++)
    summation += ((float3*)eyeData.d_compoundBuffer)[eyeData.ommatidialCount*i + ommatidiumIndex];
  return summation;
}

/*
 *  Similar to 'single_dimension_fast', but doesn't even average all
 *  samples, instead giving just the raw ommatidial data, each column
 *  an ommatidium, each row a sample in that ommatidium.
 */
extern "C" __global__ void __raygen__compound_projection_raw_ommatidial_samples()
{
  CompoundEyePosedData* posedData = (CompoundEyePosedData*)optixGetSbtDataPointer();
  const uint3 launch_idx          = optixGetLaunchIndex();
  const uint3 launch_dims         = optixGetLaunchDimensions();
  const CompoundEyeData& eyeData  = posedData->specializedData;

  // Break if this is not a pixel to render:
  if(launch_idx.y >= eyeData.samplesPerOmmatidium || launch_idx.x >= eyeData.ommatidialCount)
    return;

  // Set the colour based on the ommatidia this pixel represents
  const uint32_t image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
  float3 pixel = ((float3*)eyeData.d_compoundBuffer)[eyeData.ommatidialCount*launch_idx.y + launch_idx.x];
  params.frame_buffer[image_index] = make_color(pixel);
}

/*
 *  Projects the compound view to the display in the form of a
 *  single-dimensional vector scaled  to fit the display
 */
extern "C" __global__ void __raygen__compound_projection_single_dimension()
{
  CompoundEyePosedData* posedData = (CompoundEyePosedData*)optixGetSbtDataPointer();
  const uint3  launch_idx      = optixGetLaunchIndex();
  const uint3  launch_dims     = optixGetLaunchDimensions();
  const size_t ommatidialCount = posedData->specializedData.ommatidialCount;

  // Scale the x coordinate by the number of ommatidia (we don't want to be reading too far off the edge of the assigned ommatidia)
  const uint32_t ommatidiumIndex = (launch_idx.x * ommatidialCount)/launch_dims.x;

  //
  // Update results
  //
  const uint32_t image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
  params.frame_buffer[image_index] = make_color(getSummedOmmatidiumData(ommatidiumIndex, posedData->specializedData));
}

/*
 *  Projects the compound view to the display in the form of a single-dimensional
 *  vector taking up only the top row of the display, with width of `ommatidialCount`
 */
extern "C" __global__ void __raygen__compound_projection_single_dimension_fast()
{
  CompoundEyePosedData* posedData = (CompoundEyePosedData*)optixGetSbtDataPointer();
  const uint3 launch_idx = optixGetLaunchIndex();

  // Break if this is not a pixel to render:
  if(launch_idx.y > 0 || launch_idx.x >= posedData->specializedData.ommatidialCount) return;

  // Set the colour based on the ommatidia this pixel represents
  params.frame_buffer[(uint32_t)launch_idx.x] = make_color(getSummedOmmatidiumData(launch_idx.x, posedData->specializedData));
}

/*
 *  Projects the positions of each ommatidium down to a unit sphere and samples the
 *  closest one, position-wise.
 */
extern "C" __global__ void __raygen__compound_projection_spherical_positionwise()
{
  CompoundEyePosedData* posedData = (CompoundEyePosedData*)optixGetSbtDataPointer();
  const uint3  launch_idx      = optixGetLaunchIndex();
  const uint3  launch_dims     = optixGetLaunchDimensions();
  const size_t ommatidialCount = posedData->specializedData.ommatidialCount;

  // Project the 2D coordinates of the display window to spherical coordinates
  const float2 d = 2.0f * make_float2(
          static_cast<float>( launch_idx.x ) / static_cast<float>( launch_dims.x ),
          static_cast<float>( launch_idx.y ) / static_cast<float>( launch_dims.y )
          ) - 1.0f;
  const float2 angles = d * make_float2(-M_PIf, M_PIf/2.0f) + make_float2(M_PIf/2.0f, 0.0f);
  const float cosY = cos(angles.y);
  const float3 unitSpherePosition= make_float3(cos(angles.x)*cosY, sin(angles.y), sin(angles.x)*cosY);

  // Finds the closest ommatidium (NOTE: This is explicitly based on the position of the base of the ommatidium)
  Ommatidium* allOmmatidia = (Ommatidium*)(posedData->specializedData.d_ommatidialArray);// List of all ommatidia
  float smallestAngle = acos(dot(allOmmatidia->relativePosition, unitSpherePosition)/(length(allOmmatidia->relativePosition)*length(unitSpherePosition)));
  float angle;
  uint32_t i, closestIndex = 0;
  for(i = 1; i<ommatidialCount; i++)
  {
    angle = acos(dot((allOmmatidia+i)->relativePosition, unitSpherePosition)/(length((allOmmatidia+i)->relativePosition)*length(unitSpherePosition)));
    if(angle < smallestAngle)
    {
      smallestAngle = angle;
      closestIndex = i;
    }
  }

  //
  // Update results
  //
  const uint32_t image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
  params.frame_buffer[image_index] = make_color(getSummedOmmatidiumData(closestIndex, posedData->specializedData));
}

/*
 *  Projects the directions of each ommatidium down to a unit sphere and samples the
 *  closest one, orientation-wise.
 */
extern "C" __global__ void __raygen__compound_projection_spherical_orientationwise()
{
  CompoundEyePosedData* posedData = (CompoundEyePosedData*)optixGetSbtDataPointer();
  const uint3  launch_idx      = optixGetLaunchIndex();
  const uint3  launch_dims     = optixGetLaunchDimensions();
  const size_t ommatidialCount = posedData->specializedData.ommatidialCount;

  // Project the 2D coordinates of the display window to spherical coordinates
  const float2 d = 2.0f * make_float2(
          static_cast<float>( launch_idx.x ) / static_cast<float>( launch_dims.x ),
          static_cast<float>( launch_idx.y ) / static_cast<float>( launch_dims.y )
          ) - 1.0f;
  const float2 angles = d * make_float2(-M_PIf, M_PIf/2.0f) + make_float2(M_PIf/2.0f, 0.0f);
  const float cosY = cos(angles.y);
  const float3 unitSpherePosition= make_float3(cos(angles.x)*cosY, sin(angles.y), sin(angles.x)*cosY);

  // Finds the closest ommatidium (NOTE: This is explicitly based on orientation)
  Ommatidium* allOmmatidia = (Ommatidium*)(posedData->specializedData.d_ommatidialArray);// List of all ommatidia
  float smallestAngle = acos(dot(allOmmatidia->relativeDirection, unitSpherePosition)/(length(allOmmatidia->relativeDirection)*length(unitSpherePosition)));
  float angle;
  uint32_t i, closestIndex = 0;
  for(i = 1; i<ommatidialCount; i++)
  {
    angle = acos(dot((allOmmatidia+i)->relativeDirection, unitSpherePosition)/(length((allOmmatidia+i)->relativeDirection)*length(unitSpherePosition)));
    if(angle < smallestAngle)
    {
      smallestAngle = angle;
      closestIndex = i;
    }
  }

  //
  // Update results
  //
  const uint32_t image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
  params.frame_buffer[image_index] = make_color(getSummedOmmatidiumData(closestIndex, posedData->specializedData));
}

/*
 *  Projects the directions of each ommatidium down to a unit sphere and samples the
 *  closest one, orientation-wise.
 */
extern "C" __global__ void __raygen__compound_projection_spherical_split_orientationwise()
{
  CompoundEyePosedData* posedData = (CompoundEyePosedData*)optixGetSbtDataPointer();
  const uint3  launch_idx      = optixGetLaunchIndex();
  const uint3  launch_dims     = optixGetLaunchDimensions();
  const size_t ommatidialCount = posedData->specializedData.ommatidialCount;

  //// Project the 2D coordinates of the display window to two sets of spherical coordinates
  // Get the 2D coordinates of the pixel
  const float2 uv = make_float2(
          static_cast<float>( launch_idx.x ) / static_cast<float>( launch_dims.x ),
          static_cast<float>( launch_idx.y ) / static_cast<float>( launch_dims.y )
          );
  //const float d = ((uv * make_float2(2.0f, 1.0f))%make_float2(1.0f))*2 -1.0f;
  const float2 scaled = uv * make_float2(2.0f, 1.0f);
  const float subtraction = scaled.x>1.0f ? 1.0f : 0.f;
  const float2 modded = make_float2(scaled.x-subtraction, scaled.y);
  const float2 d = modded*2.0 -1.0f;
  const float2 angles = d * make_float2(-M_PIf, M_PIf/2.0f) + make_float2(M_PIf/2.0f, 0.0f);
  const float cosY = cos(angles.y);
  const float3 unitSpherePosition= make_float3(cos(angles.x)*cosY, sin(angles.y), sin(angles.x)*cosY);

  // Finds the closest ommatidium (NOTE: This is explicitly based on orientation)
  // (ALSO NOTE: In this, the "split" version, those points on the positive x axis are considered only by pixels on the right,
  //             the inverse is true of those on the left)
  Ommatidium* allOmmatidia = (Ommatidium*)(posedData->specializedData.d_ommatidialArray);// List of all ommatidia
  float smallestAngle = acos(dot(allOmmatidia->relativeDirection, unitSpherePosition)/(length(allOmmatidia->relativeDirection)*length(unitSpherePosition)));
  float angle;
  uint32_t i, closestIndex = 0;
  for(i = 1; i<ommatidialCount; i++)
  {
    angle = acos(dot((allOmmatidia+i)->relativeDirection, unitSpherePosition)/(length((allOmmatidia+i)->relativeDirection)*length(unitSpherePosition)));
    if( (((allOmmatidia+i)->relativePosition.x > 0.f && uv.x > 0.5f) || ((allOmmatidia+i)->relativePosition.x < 0.f && uv.x < 0.5f))
        && angle < smallestAngle)
    {
      smallestAngle = angle;
      closestIndex = i;
    }
  }

  //
  // Update results
  //
  const uint32_t image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
  params.frame_buffer[image_index] = make_color(getSummedOmmatidiumData(closestIndex, posedData->specializedData));
}

/*
 *  Projects the directions of each ommatidium down to a unit sphere and then renders the voroni diagram
 *  with each section not showing a sample from it's ommatidium, but instead the index of the ommatidium
 *  encoded into the RGB channels (base-256)
 */
extern "C" __global__ void __raygen__compound_projection_spherical_orientationwise_ids()
{
  CompoundEyePosedData* posedData = (CompoundEyePosedData*)optixGetSbtDataPointer();
  const uint3  launch_idx      = optixGetLaunchIndex();
  const uint3  launch_dims     = optixGetLaunchDimensions();
  const size_t ommatidialCount = posedData->specializedData.ommatidialCount;

  // Project the 2D coordinates of the display window to spherical coordinates
  const float2 d = 2.0f * make_float2(
          static_cast<float>( launch_idx.x ) / static_cast<float>( launch_dims.x ),
          static_cast<float>( launch_idx.y ) / static_cast<float>( launch_dims.y )
          ) - 1.0f;
  const float2 angles = d * make_float2(-M_PIf, M_PIf/2.0f) + make_float2(M_PIf/2.0f, 0.0f);
  const float cosY = cos(angles.y);
  const float3 unitSpherePosition= make_float3(cos(angles.x)*cosY, sin(angles.y), sin(angles.x)*cosY);

  // Finds the closest ommatidium (NOTE: This is explicitly based on orientation)
  Ommatidium* allOmmatidia = (Ommatidium*)(posedData->specializedData.d_ommatidialArray);// List of all ommatidia
  float smallestAngle = acos(dot(allOmmatidia->relativeDirection, unitSpherePosition)/(length(allOmmatidia->relativeDirection)*length(unitSpherePosition)));
  float angle;
  uint32_t i, closestIndex = 0;
  for(i = 1; i<ommatidialCount; i++)
  {
    angle = acos(dot((allOmmatidia+i)->relativeDirection, unitSpherePosition)/(length((allOmmatidia+i)->relativeDirection)*length(unitSpherePosition)));
    if(angle < smallestAngle)
    {
      smallestAngle = angle;
      closestIndex = i;
    }
  }

  //
  // Update results
  //
  const uint32_t image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
  const uint8_t id_red   = closestIndex >> 24;
  const uint8_t id_green = (closestIndex >> 16) & 0xff;
  const uint8_t id_blue  = (closestIndex >> 8) & 0xff;
  const uint8_t id_alpha = closestIndex & 0xff;
  //if(image_index == 0)
  //{
  //  printf("------------------------------------------ id readouts");
  //  printf("closestIndex: %u\tRGBA: %u %u %u %u", closestIndex, id_red, id_green, id_blue, id_alpha);
  //}
  params.frame_buffer[image_index] = make_uchar4(id_red, id_green, id_blue, id_alpha);
}

/*
 *  Projects the position of each ommatidium down to a unit sphere and then renders the voroni diagram
 *  with each section not showing a sample from it's ommatidium, but instead the index of the ommatidium
 *  encoded into the RGB channels (base-256)
 */
extern "C" __global__ void __raygen__compound_projection_spherical_positionwise_ids()
{
  CompoundEyePosedData* posedData = (CompoundEyePosedData*)optixGetSbtDataPointer();
  const uint3  launch_idx      = optixGetLaunchIndex();
  const uint3  launch_dims     = optixGetLaunchDimensions();
  const size_t ommatidialCount = posedData->specializedData.ommatidialCount;

  // Project the 2D coordinates of the display window to spherical coordinates
  const float2 d = 2.0f * make_float2(
          static_cast<float>( launch_idx.x ) / static_cast<float>( launch_dims.x ),
          static_cast<float>( launch_idx.y ) / static_cast<float>( launch_dims.y )
          ) - 1.0f;
  const float2 angles = d * make_float2(-M_PIf, M_PIf/2.0f) + make_float2(M_PIf/2.0f, 0.0f);
  const float cosY = cos(angles.y);
  const float3 unitSpherePosition= make_float3(cos(angles.x)*cosY, sin(angles.y), sin(angles.x)*cosY);

  // Finds the closest ommatidium (NOTE: This is explicitly based on the position of the base of the ommatidium)
  Ommatidium* allOmmatidia = (Ommatidium*)(posedData->specializedData.d_ommatidialArray);// List of all ommatidia
  float smallestAngle = acos(dot(allOmmatidia->relativePosition, unitSpherePosition)/(length(allOmmatidia->relativePosition)*length(unitSpherePosition)));
  float angle;
  uint32_t i, closestIndex = 0;
  for(i = 1; i<ommatidialCount; i++)
  {
    angle = acos(dot((allOmmatidia+i)->relativePosition, unitSpherePosition)/(length((allOmmatidia+i)->relativePosition)*length(unitSpherePosition)));
    if(angle < smallestAngle)
    {
      smallestAngle = angle;
      closestIndex = i;
    }
  }

  //
  // Update results
  //
  const uint32_t image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
  const uint8_t id_red   = closestIndex >> 24;
  const uint8_t id_green = (closestIndex >> 16) & 0xff;
  const uint8_t id_blue  = (closestIndex >> 8) & 0xff;
  const uint8_t id_alpha = closestIndex & 0xff;
  params.frame_buffer[image_index] = make_uchar4(id_red, id_green, id_blue, id_alpha);
}

//------------------------------------------------------------------------------
//
//  Ommatidial Ray Generation Programs
//
//------------------------------------------------------------------------------

__device__ inline float3 rotatePoint(const float3 point, const float angle, const float3 axis)
{
  return (cos(angle)*point + sin(angle)*cross(axis, point) + (1 - cos(angle))*dot(axis, point)*axis);
}
__device__ float3 generateOffsetRay( const float ommatidialAxisAngle, const float splayAngle, const float3 ommatidialAxis)
{
    //// Rotate the ommatidial axis about a perpendicular vector by splay angle
    float3 perpAxis = cross(make_float3(0.0f, 1.0f, 0.0f), ommatidialAxis);
    // Check that the perpAxis isn't zero (because ommatidialAxis was pointing directly up) (could probably be done with a memcmp for speed)
    perpAxis = (perpAxis.x + perpAxis.y + perpAxis.z == 0.0f) ? make_float3(0.0f, 0.0f, 1.0f) : normalize(perpAxis);
    // Rotate by the splay angle
    const float3 splayedAxis = rotatePoint(ommatidialAxis, splayAngle, perpAxis);
    //// Rotate the new axis around the original ommatidial axis by the ommatidialAxisAngle
    return rotatePoint(splayedAxis, ommatidialAxisAngle, ommatidialAxis);
}

extern "C" __global__ void __raygen__ommatidium()
{
  const uint3 launch_idx = optixGetLaunchIndex();
  const uint3 launch_dims = optixGetLaunchDimensions();
  const uint32_t ommatidialIndex = launch_idx.x;
  const int id = launch_dims.x * launch_idx.y + launch_idx.x;
  const RecordPointer* recordPointer = (RecordPointer*)optixGetSbtDataPointer();// Gets the compound record, which points to the current camera's record.
  const CompoundEyePosedData posedData = ((CompoundEyePosedDataRecord*)(recordPointer->d_record))->data; // Contains the actual posed eye data

  Ommatidium* allOmmatidia = (Ommatidium*)(posedData.specializedData.d_ommatidialArray);// List of all ommatidia
  Ommatidium ommatidium = *(allOmmatidia + ommatidialIndex);// This ommatidium

  // Get the relative direction of the ommatidial axis
  const float3 relativeOmmatidialAxis = ommatidium.relativeDirection;
  const float3 relativeOmmatidialPosition = ommatidium.relativePosition;

  curandState localState; // A local copy of the cuRand state (to be) stored in shared memory
  curandState& sharedState = ((curandState*)(posedData.specializedData.d_randomStates))[id]; // A reference to the original cuRand state stored in shared memory
  if(!posedData.specializedData.randomsConfigured)
  {
    curand_init(42, id, 0, &localState); // Initialize the state if it needs to be
  }else{
    localState = sharedState; // Pull down the random state of this ommatidium
  }

  // Calculate the s.d. to scale a standard normal random value up to so that it matches the acceptance angle
  const float standardDeviation = ommatidium.acceptanceAngleRadians/FWHM_SD_RATIO;
  float splayAngle = curand_normal(&localState) * standardDeviation;// Angle away from the ommatidial axis
  float ommatidialAxisAngle = curand_uniform(&localState)*M_PIf;// Angle around the ommatidial axis (note that it only needs to rotate through 180 degrees because splayAngle can be negative)

  // Copy the RNG state back into the buffer for use next time
  sharedState = localState;

  // Generate a pair of angles away from the ommatidial axis
  const float3 relativeDir = generateOffsetRay(ommatidialAxisAngle, splayAngle, relativeOmmatidialAxis);

  // Move the start of the ray into the eye along the ommatidial axis by focalPointOffset
  const float3 relativePos = relativeOmmatidialPosition - normalize(relativeOmmatidialAxis) * ommatidium.focalPointOffset;

  // Transform ray information into world-space
  const float3 ray_origin = posedData.position + posedData.localSpace.xAxis*relativePos.x
                                               + posedData.localSpace.yAxis*relativePos.y
                                               + posedData.localSpace.zAxis*relativePos.z;
  const float3 ray_direction = posedData.localSpace.xAxis * relativeDir.x
                             + posedData.localSpace.yAxis * relativeDir.y
                             + posedData.localSpace.zAxis * relativeDir.z;

  // Transmit the ray
  globalParameters::PayloadRadiance payload;
  payload.result = make_float3( 0.0f );
  payload.importance = 1.0f;
  payload.depth = 0.0f;

  traceRadiance(
          params.handle,
          ray_origin,
          ray_direction,
          ommatidium.focalPointOffset, // tmin, the surface of the top of the ommatidial lens
          1e16f,  // tmax
          &payload );

  //
  // Add results to this eye's compound buffer
  // This mixes in the feedback from each sample ray with respect to the it's position in the rendering volume.
  // For instance, if each ommatidium is to make 20 samples then each launch of this shader is one sample and only
  // contributes 0.05/1 to the final colour in the compound buffer.
  ((float3*)posedData.specializedData.d_compoundBuffer)[id] = payload.result*(1.0f/posedData.specializedData.samplesPerOmmatidium); // Scale it down as these will be summed in the projection shader
}


//------------------------------------------------------------------------------
//
//  Miss programs
//
//------------------------------------------------------------------------------

extern "C" __global__ void __miss__default_background()
{
    const float3 dir = normalize(optixGetWorldRayDirection());
    setPayloadResult(make_float3((atan2(dir.z, dir.x)+M_PIf)/(M_PIf*2.0f), (asin(dir.y)+M_PIf/2.0f)/M_PIf, 0.0f));
    const float border = 0.01;
    if(abs(dir.x) < border || abs(dir.y) < border || abs(dir.z) < border)
      setPayloadResult(make_float3(0.0f));
}

extern "C" __global__ void __miss__simple_sky()
{
    const float3 dir = normalize(optixGetWorldRayDirection());
    const float mix = min(max(0.0f, (asin(dir.y)*2.0f)/M_PIf), 1.0f);
    const float3 upper = make_float3(1.0f, 31.0f, 117.0f)/255.0f;
    const float3 lower = make_float3(143.0f, 179.0f, 203.0f)/255.0f * 0.8f;
    setPayloadResult( lower*(1.0f-mix) + upper*mix );
}

//extern "C" __global__ void __miss__sky_and_grass()
//{
//    const float3 dir = normalize(optixGetWorldRayDirection());
//    const float mix = min(max(0.0f, (asin(dir.y)*2.0f)/M_PIf), 1.0f);
//    const float3 upper = make_float3(1.0f, 31.0f, 117.0f)/255.0f;
//    const float3 lower = make_float3(143.0f, 179.0f, 203.0f)/255.0f * 0.8f;
//    setPayloadResult( lower*(1.0f-mix) + upper*mix );
//}

//------------------------------------------------------------------------------
//
//  Old Hit Programs
//
//------------------------------------------------------------------------------

extern "C" __global__ void __closesthit__occlusion()
{
    setPayloadOcclusion( true );
}


extern "C" __global__ void __closesthit__radiance()
{
    //setPayloadResult( make_float3(1.0f));
    const globalParameters::HitGroupData* hit_group_data = reinterpret_cast<globalParameters::HitGroupData*>( optixGetSbtDataPointer() );
    const LocalGeometry          geom           = getLocalGeometry( hit_group_data->geometry_data );

    //
    // Retrieve material data
    //
    float3 base_color = make_float3( hit_group_data->material_data.pbr.base_color );

    if(geom.UC)   // TODO: UNFIX
    {
//      //base_color *= linearize(make_float3(geom.C.x, geom.C.y, geom.C.z));
//      //base_color = geom.C;//make_float3(geom.C.x, geom.C.y, geom.C.z);
//      //base_color *= linearize(geom.C);
//      //base_color = linearize(geom.C);
      base_color = linearize(make_float3(geom.C.x, geom.C.y, geom.C.z));
//      //base_color = make_float3(1.0f, 0.0f, 0.0f);//make_float3(geom.C.x, geom.C.y, geom.C.z);
//        base_color *= linearize( make_float3(
//                    tex2D<float4>( hit_group_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y )
//                    ) );
    }else if( hit_group_data->material_data.pbr.base_color_tex ){
        base_color = linearize( make_float3(
                    tex2D<float4>( hit_group_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y )
                    ) );
    }

    if(!params.lighting)
    {
      setPayloadResult( base_color);
      return;
    }

    float metallic  = hit_group_data->material_data.pbr.metallic;
    float roughness = hit_group_data->material_data.pbr.roughness;
    float4 mr_tex = make_float4( 1.0f );
    if( hit_group_data->material_data.pbr.metallic_roughness_tex )
        // MR tex is (occlusion, roughness, metallic )
        mr_tex = tex2D<float4>( hit_group_data->material_data.pbr.metallic_roughness_tex, geom.UV.x, geom.UV.y );
    roughness *= mr_tex.y;
    metallic  *= mr_tex.z;


    //
    // Convert to material params
    //
    const float  F0         = 0.04f;
    const float3 diff_color = base_color*( 1.0f - F0 )*( 1.0f - metallic );
    const float3 spec_color = lerp( make_float3( F0 ), base_color, metallic );
    const float  alpha      = roughness*roughness;

    //
    // compute direct lighting
    //

    float3 N = geom.N;
    if( hit_group_data->material_data.pbr.normal_tex )
    {
        const float4 NN = 2.0f*tex2D<float4>( hit_group_data->material_data.pbr.normal_tex, geom.UV.x, geom.UV.y ) - make_float4(1.0f);
        N = normalize( NN.x*normalize( geom.dpdu ) + NN.y*normalize( geom.dpdv ) + NN.z*geom.N );
    }

    float3 result = make_float3( 0.0f );

    for( int i = 0; i < params.lights.count; ++i )
    {
        Light::Point light = params.lights[i];

        // TODO: optimize
        const float  L_dist  = length( light.position - geom.P );
        const float3 L       = ( light.position - geom.P ) / L_dist;
        const float3 V       = -normalize( optixGetWorldRayDirection() );
        const float3 H       = normalize( L + V );
        const float  N_dot_L = dot( N, L );
        const float  N_dot_V = dot( N, V );
        const float  N_dot_H = dot( N, H );
        const float  V_dot_H = dot( V, H );

        if( N_dot_L > 0.0f && N_dot_V > 0.0f )
        {
            const float tmin     = 0.001f;          // TODO
            const float tmax     = L_dist - 0.001f; // TODO
            const bool  occluded = traceOcclusion( params.handle, geom.P, L, tmin, tmax );
            if( !occluded )
            {
                const float3 F     = schlick( spec_color, V_dot_H );
                const float  G_vis = vis( N_dot_L, N_dot_V, alpha );
                const float  D     = ggxNormal( N_dot_H, alpha );

                const float3 diff = ( 1.0f - F )*diff_color / M_PIf;
                const float3 spec = F*G_vis*D;

                result += light.color*light.intensity*N_dot_L*( diff + spec );
            }
        }
    }
    // TODO: add debug viewing mode that allows runtime switchable views of shading params, normals, etc
    //result = make_float3( roughness );
    //result = N*0.5f + make_float3( 0.5f );
    //result = geom.N*0.5f + make_float3( 0.5f );
    setPayloadResult( result );
}
