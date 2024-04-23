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

#include <optix.h>

#include <sutil/Matrix.h>
#include <sutil/Preprocessor.h>
#include <sutil/vec_math.h>
#include <cuda/BufferView.h>
#include <cuda/GeometryData.h>
#include <cuda/util.h>


struct LocalGeometry
{
    float3 P;
    float3 N;
    float3 Ng;
    float2 UV;
    float3 dndu;
    float3 dndv;
    float3 dpdu;
    float3 dpdv;
    float4 C; // Vertex colour
    bool UC;  // Use vertex colour?
};


SUTIL_HOSTDEVICE LocalGeometry getLocalGeometry( const GeometryData& geometry_data )
{
    LocalGeometry lgeom;
    switch( geometry_data.type )
    {
        case GeometryData::TRIANGLE_MESH:
        {
            const GeometryData::TriangleMesh& mesh_data = geometry_data.triangle_mesh;

            const uint32_t prim_idx = optixGetPrimitiveIndex();
            const float2   barys    = optixGetTriangleBarycentrics();

            uint3 tri = make_uint3(0u, 0u, 0u);
            if( mesh_data.indices.elmt_byte_size == 4 )
            {
                const uint3* indices = reinterpret_cast<uint3*>( mesh_data.indices.data );
                tri = indices[ prim_idx ];
            }
            else
            {
                const uint16_t* indices = reinterpret_cast<uint16_t*>( mesh_data.indices.data );
                const uint16_t idx0 = indices[ prim_idx*3+0 ];
                const uint16_t idx1 = indices[ prim_idx*3+1 ];
                const uint16_t idx2 = indices[ prim_idx*3+2 ];
                tri = make_uint3( idx0, idx1, idx2 );
            }

            const float3 P0 = mesh_data.positions[ tri.x ];
            const float3 P1 = mesh_data.positions[ tri.y ];
            const float3 P2 = mesh_data.positions[ tri.z ];
            lgeom.P = ( 1.0f-barys.x-barys.y)*P0 + barys.x*P1 + barys.y*P2;
            lgeom.P = optixTransformPointFromObjectToWorldSpace( lgeom.P );

            // Set UV texture coordinates
            float2 UV0, UV1, UV2;
            if( mesh_data.texcoords )
            {
                UV0 = mesh_data.texcoords[ tri.x ];
                UV1 = mesh_data.texcoords[ tri.y ];
                UV2 = mesh_data.texcoords[ tri.z ];
                lgeom.UV = ( 1.0f-barys.x-barys.y)*UV0 + barys.x*UV1 + barys.y*UV2;
            }
            else
            {
                UV0 = make_float2( 0.0f, 0.0f );
                UV1 = make_float2( 0.0f, 1.0f );
                UV2 = make_float2( 1.0f, 0.0f );
                lgeom.UV = barys;
            }

            // Set vertex colour coordinates
            float4 Cf0, Cf1, Cf2;
            if( mesh_data.dev_color_type != -1 )
            {

              switch(mesh_data.dev_color_type)
              {
                case 5121: // UNSIGNED BYTE
                  uchar4 Cc0, Cc1, Cc2;
                  Cc0 = mesh_data.dev_colors_uc4[ tri.x ];
                  Cc1 = mesh_data.dev_colors_uc4[ tri.y ];
                  Cc2 = mesh_data.dev_colors_uc4[ tri.z ];
                  Cf0 = make_float4( Cc0.x, Cc0.y, Cc0.z, Cc0.w );
                  Cf1 = make_float4( Cc1.x, Cc1.y, Cc1.z, Cc1.w );
                  Cf2 = make_float4( Cc2.x, Cc2.y, Cc2.z, Cc2.w );
                  Cf0 /= 255.0f;
                  Cf1 /= 255.0f;
                  Cf2 /= 255.0f;
                  break;
                case 5123: // UNSIGNED SHORT
                  ushort4 Cs0, Cs1, Cs2;
                  Cs0 = mesh_data.dev_colors_us4[ tri.x ];
                  Cs1 = mesh_data.dev_colors_us4[ tri.y ];
                  Cs2 = mesh_data.dev_colors_us4[ tri.z ];
                  Cf0 = make_float4( Cs0.x, Cs0.y, Cs0.z, Cs0.w );
                  Cf1 = make_float4( Cs1.x, Cs1.y, Cs1.z, Cs1.w );
                  Cf2 = make_float4( Cs2.x, Cs2.y, Cs2.z, Cs2.w );
                  Cf0 /= 65535.0f;
                  Cf1 /= 65535.0f;
                  Cf2 /= 65535.0f;
                  break;
                case 5126: // FLOAT
                  if (mesh_data.color_container == 3) {
                    // translate from vec3 of float to vec4 of float:
                    float3 Cf3_0 = mesh_data.dev_colors_f3[ tri.x ];
                    float3 Cf3_1 = mesh_data.dev_colors_f3[ tri.y ];
                    float3 Cf3_2 = mesh_data.dev_colors_f3[ tri.z ];
                    Cf0 = make_float4( Cf3_0.x, Cf3_0.y, Cf3_0.z, 1.0f );
                    Cf1 = make_float4( Cf3_1.x, Cf3_1.y, Cf3_1.z, 1.0f );
                    Cf2 = make_float4( Cf3_2.x, Cf3_2.y, Cf3_2.z, 1.0f );
                  } else if (mesh_data.color_container == 4) {
                    Cf0 = mesh_data.dev_colors_f4[ tri.x ];
                    Cf1 = mesh_data.dev_colors_f4[ tri.y ];
                    Cf2 = mesh_data.dev_colors_f4[ tri.z ];
                  } else {
                    // unhandled color container size
                    Cf0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    Cf1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                    Cf2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                  }
                  break;
                default:
                  Cf0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                  Cf1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                  Cf2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                  break;
              }

              lgeom.C = ( 1.0f-barys.x-barys.y)*Cf0 + barys.x*Cf1 + barys.y*Cf2;
              //lgeom.C = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
              lgeom.UC = true;
            }
            else
            {
              lgeom.C = make_float4( 1.0f, 0.0f, 0.0f,1.0f);
              lgeom.UC = false;
            }

            lgeom.Ng = normalize( cross( P1-P0, P2-P0 ) );
            lgeom.Ng = optixTransformNormalFromObjectToWorldSpace( lgeom.Ng );

            float3 N0, N1, N2;
            if( mesh_data.normals )
            {
                N0 = mesh_data.normals[ tri.x ];
                N1 = mesh_data.normals[ tri.y ];
                N2 = mesh_data.normals[ tri.z ];
                lgeom.N = ( 1.0f-barys.x-barys.y)*N0 + barys.x*N1 + barys.y*N2;
                lgeom.N = normalize( optixTransformNormalFromObjectToWorldSpace( lgeom.N ) );
            }
            else
            {
                lgeom.N = N0 = N1 = N2 = lgeom.Ng;
            }

            const float du1 = UV0.x - UV2.x;
            const float du2 = UV1.x - UV2.x;
            const float dv1 = UV0.y - UV2.y;
            const float dv2 = UV1.y - UV2.y;

            const float3 dp1 = P0 - P2;
            const float3 dp2 = P1 - P2;

            const float3 dn1 = N0 - N2;
            const float3 dn2 = N1 - N2;

            const float det = du1*dv2 - dv1*du2;

            const float invdet = 1.f / det;
            lgeom.dpdu = ( dv2 * dp1 - dv1 * dp2) * invdet;
            lgeom.dpdv = (-du2 * dp1 + du1 * dp2) * invdet;
            lgeom.dndu = ( dv2 * dn1 - dv1 * dn2) * invdet;
            lgeom.dndu = (-du2 * dn1 + du1 * dn2) * invdet;


            break;
        }
        case GeometryData::SPHERE:
        {
            break;
        }
        default: break;
    }


    return lgeom;
}
