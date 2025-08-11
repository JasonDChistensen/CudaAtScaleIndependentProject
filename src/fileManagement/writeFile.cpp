#include <iostream>
#include <fstream>
#include <vector>

void writeFile(std::string fileName, const std::vector<float>& data)
{
    FILE * pFile;
    
    // Open the file
    pFile = fopen ( fileName.c_str() , "wb" );
    if (pFile==NULL)
    {
       printf("Error opening file:%s\n", fileName.c_str());
       exit (1);
    }

    fwrite (data.data() , sizeof(float), data.size(), pFile);
    fclose (pFile);
}