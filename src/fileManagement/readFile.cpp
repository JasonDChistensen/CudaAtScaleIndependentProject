#include <iostream>
#include <fstream>
#include <vector>

std::vector<float> readFile(std::string fileName)
{
  FILE * pFile;
  long lSize;
  size_t result;

  // Open the file
  pFile = fopen ( fileName.c_str() , "rb" );
  if (pFile==NULL) {fputs ("File error",stderr); exit (1);}

  // obtain file size:
  fseek (pFile , 0 , SEEK_END);
  lSize = ftell (pFile);
  rewind (pFile);

  // Create the buffer to hold the data
  const long sizeInFloats = lSize/sizeof(float);
  std::vector<float> buffer(sizeInFloats);
  
  // Read the data into the buffer
  result = fread (buffer.data(), 1, lSize, pFile);
  if (result != lSize) {fputs ("Reading error",stderr); exit (3);}

  fclose (pFile);
  return buffer;
}
