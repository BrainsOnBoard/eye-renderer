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

extern "C"
{
__constant__ globalParameters::LaunchParams params;
}


//------------------------------------------------------------------------------
//
// GGX/smith shading helpers
// TODO: move into header so can be shared by path tracer and bespoke renderers
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
    optixSetPayload_0( float_as_int( p.x ) );
    optixSetPayload_1( float_as_int( p.y ) );
    optixSetPayload_2( float_as_int( p.z ) );
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
//  Ommatidial Ray Generation Programs
//
//------------------------------------------------------------------------------

extern "C" __global__ void __raygen__compound_projection_single_dimension()
{
    CompoundEyePosedData* posedData = (CompoundEyePosedData*)optixGetSbtDataPointer();
    const uint3  launch_idx      = optixGetLaunchIndex();
    const uint3  launch_dims     = optixGetLaunchDimensions();
    const uint32_t eyeIndex      = posedData->specializedData.eyeIndex;
    const uint32_t compWidth     = params.compoundBufferWidth;
    const uint32_t compHeight    = params.compoundBufferHeight;
    const size_t ommatidialCount = posedData->specializedData.ommatidialCount;

    // Scale the x coordinate by the number of ommatidia (we don't want to be reading too far off the edge of the assigned ommatidia)
    const uint32_t ommatidiumIndex = (launch_idx.x * ommatidialCount)/launch_dims.x;

    //
    // Update results
    //
    const uint32_t image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
    params.frame_buffer[image_index] = params.compound_buffer[eyeIndex*compWidth + ommatidiumIndex];
}

extern "C" __global__ void __raygen__compound_projection_spherical_positionwise()
{
    CompoundEyePosedData* posedData = (CompoundEyePosedData*)optixGetSbtDataPointer();
    const uint3  launch_idx      = optixGetLaunchIndex();
    const uint3  launch_dims     = optixGetLaunchDimensions();
    const uint32_t eyeIndex      = posedData->specializedData.eyeIndex;
    const uint32_t compWidth     = params.compoundBufferWidth;
    const uint32_t compHeight    = params.compoundBufferHeight;
    const size_t ommatidialCount = posedData->specializedData.ommatidialCount;

    // Project the 2D coordinates of the display window to spherical coordinates
    const float2 d = 2.0f * make_float2(
            static_cast<float>( launch_idx.x ) / static_cast<float>( launch_dims.x ),
            static_cast<float>( launch_idx.y ) / static_cast<float>( launch_dims.y )
            ) - 1.0f;
    const float2 angles = d * make_float2(-M_PIf, M_PIf/2.0f) + make_float2(M_PIf/2.0f, 0.0f);
    const float cosY = cos(angles.y);
    const float3 unitSpherePosition= make_float3(cos(angles.x)*cosY, sin(angles.y), sin(angles.x)*cosY);

    // Finds the closest ommatidium (NOTE: This is explicitly based on position)
    Ommatidium* allOmmatidia = (Ommatidium*)(posedData->specializedData.d_ommatidialArray);// List of all ommatidia
    float dx = allOmmatidia->relativePosition.x - unitSpherePosition.x;
    float dy = allOmmatidia->relativePosition.y - unitSpherePosition.y;
    float dz = allOmmatidia->relativePosition.z - unitSpherePosition.z;
    float closestDistance = dx*dx+dy*dy+dz*dz;
    float dist;
    uint32_t i, closestIndex = 0;
    for(i = 1; i<ommatidialCount; i++)
    {
      dx = (allOmmatidia + i)->relativePosition.x - unitSpherePosition.x;
      dy = (allOmmatidia + i)->relativePosition.y - unitSpherePosition.y;
      dz = (allOmmatidia + i)->relativePosition.z - unitSpherePosition.z;
      dist = dx*dx+dy*dy+dz*dz;
      if(dist <closestDistance)
      {
        closestDistance = dist;
        closestIndex = i;
      }
    }

    //
    // Update results
    //
    const uint32_t image_index  = launch_idx.y * launch_dims.x + launch_idx.x;
    params.frame_buffer[image_index] = params.compound_buffer[eyeIndex*compWidth + closestIndex];
}

extern "C" __global__ void __raygen__ommatidium()
{
  const uint3 launch_idx = optixGetLaunchIndex();
  const uint3 launch_dims = optixGetLaunchDimensions();
  const uint32_t eyeIndex = launch_idx.y;
  const uint32_t ommatidialIndex = launch_idx.x;
  CompoundEyeCollectionData* eyeCollection = (CompoundEyeCollectionData*)optixGetSbtDataPointer();

  //if(threadIdx.x == 1)
  //{
  //  printf("%i eyes found\n", eyeCollection->eyeCount);
  //  CUdeviceptr* eyes = (CUdeviceptr*)eyeCollection->d_compoundEyes;
  //  for(uint32_t i = 0; i<eyeCollection->eyeCount; i++)
  //  {
  //    CompoundEyePosedDataRecord* eyeRecord = (CompoundEyePosedDataRecord*)(*(eyes + i));
  //    printf(" Eye pointer   : %p\n", eyeRecord);
  //    CompoundEyePosedData eyeData = eyeRecord->data;
  //    printf("   Eye %i position: (%f, %f, %f)\n", i, eyeData.position.x, eyeData.position.y, eyeData.position.z);
  //  }

  //}

  CUdeviceptr* eyes = (CUdeviceptr*)eyeCollection->d_compoundEyes;// List of all eye records
  CompoundEyePosedDataRecord* eyeRecord = (CompoundEyePosedDataRecord*)(*(eyes + eyeIndex)); // This eye record
  CompoundEyePosedData eyeData = eyeRecord->data;// This eye record's data

  if(ommatidialIndex >= eyeData.specializedData.ommatidialCount)
    return;// Exit if you're going to be trying to reference eyes that don't exist

  Ommatidium* allOmmatidia = (Ommatidium*)(eyeData.specializedData.d_ommatidialArray);// List of all ommatidia
  Ommatidium ommatidium = *(allOmmatidia + ommatidialIndex);// This ommatidium
  const float3 relativePos = ommatidium.relativePosition;
  const float3 relativeDir = ommatidium.relativeDirection;

  // Transform ray information into world-space
  const float3 ray_origin = eyeData.position + eyeData.localSpace.xAxis*relativePos.x
                                             + eyeData.localSpace.yAxis*relativePos.y
                                             + eyeData.localSpace.zAxis*relativePos.z;
  const float3 ray_direction = eyeData.localSpace.xAxis * relativeDir.x
                             + eyeData.localSpace.yAxis * relativeDir.y
                             + eyeData.localSpace.zAxis * relativeDir.z;

  //if(eyeIndex == 0)
  //{
  //  printf("RelativeDir: (%.2f, %.2f, %.2f)\n", relativeDir.x, relativeDir.y, relativeDir.z);
  //  printf("  World dir: (%.2f, %.2f, %.2f)\n", ray_direction.x, ray_direction.y, ray_direction.z);
  //  printf("      xAxis: (%.2f, %.2f, %.2f)\n", eyeData.localSpace.xAxis.x, eyeData.localSpace.xAxis.y, eyeData.localSpace.xAxis.z);
  //}

  // Transmit the ray
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
  const uint32_t compoundIndex = eyeIndex * launch_dims.x + ommatidialIndex;
  params.compound_buffer[compoundIndex] = make_color(payload.result);
}

//------------------------------------------------------------------------------
//
//  Miss programs
//
//------------------------------------------------------------------------------

extern "C" __global__ void __miss__constant_radiance()
{
    //setPayloadResult( params.miss_color );
    const float3 dir = normalize(optixGetWorldRayDirection());
    setPayloadResult(make_float3((atan2(dir.z, dir.x)+M_PIf)/(M_PIf*2.0f), (asin(dir.y)+M_PIf/2.0f)/(M_PIf), 0.0f));
    const float border = 0.01;
    if(abs(dir.x) < border || abs(dir.y) < border || abs(dir.z) < border)
      setPayloadResult(make_float3(0.0f));
}

//------------------------------------------------------------------------------
//
//  Hit Programs
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
    if( hit_group_data->material_data.pbr.base_color_tex )
        base_color *= linearize( make_float3(
                    tex2D<float4>( hit_group_data->material_data.pbr.base_color_tex, geom.UV.x, geom.UV.y )
                    ) );

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
