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
#pragma once

#include <cuda/BufferView.h>


struct GeometryData
{
    enum Type
    {
        TRIANGLE_MESH = 0,
        SPHERE        = 1
    };


    struct TriangleMesh
    {
        GenericBufferView    indices;
        BufferView<float3>   positions;
        BufferView<float3>   normals;
        BufferView<float2>   texcoords;
        BufferView<float3>  dev_colors_f3;
        BufferView<float4>  dev_colors_f4;
        BufferView<ushort4> dev_colors_us4;
        BufferView<uchar4>  dev_colors_uc4;
        // Holds the type of the color data, which may be float or unsigned char/short
        int dev_color_type = -1;
        // Holds a number referring to the format of the color buffer, which may be vec3 of float,
        // vec4 of float, vec4 of unsigned char, vec4 of unsigned short (perhaps also vec3 of
        // unsigned char and vec3 of unsigned short, but these are not supported at time of
        // writing).
        int color_container = -1;
    };


    struct Sphere
    {
        float3 center;
        float  radius;
    };


    Type  type;

    union
    {
        TriangleMesh triangle_mesh;
        Sphere        sphere;
    };
};
