#include <gtest/gtest.h>
#include "readFile.hh"

TEST(FILEMANAGEMENT, readFile)
{
    std::vector<float> data = readFile("./vectors/fileRead/minus10to10.bin");
    ASSERT_EQ(data.size(), 21);

    float expected_value = -10.0;
    for(const auto& d : data)
    {
        EXPECT_NEAR(d, expected_value, 1.0e-6);
        expected_value += 1.0;
    }
}
