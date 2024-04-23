#ifndef LIB_EYE_RENDERER_3_H
#define LIB_EYE_RENDERER_3_H
#include <cstddef>
#include <vector_types.h>

// A simplified ommatidium object, to make it easier to
// transfer ommatidial information from external API users.
struct OmmatidiumPacket
{
  float posX,posY,posZ;
  float dirX,dirY,dirZ;
  float acceptanceAngle;
  float focalpointOffset;
};

extern "C"
{
  // Configuration
  void setVerbosity(bool v);                // turns on/off the '[PyEye]' debug outputs
  void loadGlTFscene(const char* filepath); // Loads a given gltf file
  void stop(void);                          // Stops the eyeRenderer in a slightly more elegant way
  void setRenderSize(int w, int h);         // Sets the width an height of the rendering frame
  double renderFrame(void);                 // Actually renders the frame, returns the time it took to render the frame (in ms)
  void displayFrame(void);                  // Displays the rendered frame to the open GL display window
  void saveFrameAs(const char* ppmFilename);// Saves the frame (must be a .ppm filename)
  unsigned char* getFramePointer(void);     // Retrieves a pointer to the frame data

  // Camera control
  size_t getCameraCount(void);
  void nextCamera(void);
  void previousCamera(void);
  size_t getCurrentCameraIndex(void);
  const char* getCurrentCameraName();
  void gotoCamera(int index);
  bool gotoCameraByName(char* name);
  void setCameraPosition(float x, float y, float z);
  void getCameraPosition(float& x, float& y, float& z);
  //void pointCameraAt
  void setCameraLocalSpace(float lxx, float lxy, float lxz,
                           float lyx, float lyy, float lyz,
                           float lzx, float lzy, float lzz);
  void rotateCameraAround(float angle, float axisX, float axisY, float axisZ);
  void rotateCameraLocallyAround(float angle, float axisX, float axisY, float axisZ);
  void translateCamera(float x, float y, float z);
  void translateCameraLocally(float x, float y, float z);
  // Resets the translation and rotation of the current camera
  void resetCameraPose();
  // Rotates the camera around rot[X,Y,Z] around world axes and then sets translation to pos[X,Y,Z]
  void setCameraPose(float posX, float posY, float posZ, float rotX, float rotY, float rotZ);

  // Compound-specific
  //int getOmmatidialCameraCount(void);
  bool isCompoundEyeActive(void);
  void setCurrentEyeSamplesPerOmmatidium(int s);// Changes the current eye samples per ommatidium. WARNING: This resets the random seed values. A render must be called to regenerate them, this will take significantly longer than a frame render.
  int  getCurrentEyeSamplesPerOmmatidium(void);// Returns the current eye samples per ommatidium
  void changeCurrentEyeSamplesPerOmmatidiumBy(int s);// Changes the current eye samples per ommatidium. WARNING: This resets the random seed values. A render must be called to regenerate them, this will take significantly longer than a frame render.
  size_t getCurrentEyeOmmatidialCount(void);// Returns the number of ommatidia in this eye
  // void setOmmatidia(size_t count, Ommatidium* omms); // Sets the ommatidia for the eye
  void setOmmatidia(OmmatidiumPacket* omms, size_t count); // Sets the ommatidia for the current eye
  const char* getCurrentEyeDataPath(void);
  void setCurrentEyeShaderName(char* name); // Sets the compound projection shader the current eye is using

  // Scene manipulation
  bool isInsideHitGeometry(float x, float y, float z, char* name); // tests whether a point is within a named piece of hit geometry
  float3 getGeometryMaxBounds(char* name); // Returns the maximal bounds of a geometry element, specified by name.
  float3 getGeometryMinBounds(char* name); // Returns the minimal bounds of a geometry element, specified by name.
}

void *getWindowPointer(); // This is a little janky, but it's probably okay, as we're avoiding a load of imports


#endif
