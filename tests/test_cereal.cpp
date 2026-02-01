/**
 * @file test_cereal.cpp
 * @brief Cereal序列化库测试
 * @details 测试二进制序列化、JSON序列化等功能
 */

#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/vector.hpp>
#include <cereal/types/map.hpp>
#include <cereal/types/string.hpp>

/**
 * @brief 测试数据结构
 */
struct TestData
{
    int id;
    std::string name;
    std::vector<float> values;

    template<class Archive>
    void serialize(Archive& archive)
    {
        archive(id, name, values);
    }

    bool operator==(const TestData& other) const
    {
        return id == other.id && name == other.name && values == other.values;
    }
};

/**
 * @brief 测试二进制序列化
 * @return 测试是否通过
 */
bool test_binary_serialization()
{
    std::cout << "[cereal] 测试二进制序列化..." << std::endl;

    try
    {
        // 创建测试数据
        TestData original;
        original.id = 42;
        original.name = "Test Object";
        original.values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

        // 序列化到内存
        std::stringstream ss;
        {
            cereal::BinaryOutputArchive archive(ss);
            archive(original);
        }

        std::cout << "序列化数据大小: " << ss.str().size() << " 字节" << std::endl;

        // 从内存反序列化
        TestData loaded;
        {
            cereal::BinaryInputArchive archive(ss);
            archive(loaded);
        }

        // 验证数据
        if (original != loaded)
        {
            std::cerr << "反序列化数据不匹配" << std::endl;
            return false;
        }

        std::cout << "[cereal] 二进制序列化测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "二进制序列化测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试JSON序列化
 * @return 测试是否通过
 */
bool test_json_serialization()
{
    std::cout << "[cereal] 测试JSON序列化..." << std::endl;

    try
    {
        // 创建测试数据
        std::map<std::string, int> scores;
        scores["Alice"] = 95;
        scores["Bob"] = 87;
        scores["Charlie"] = 92;

        // 序列化为JSON
        std::stringstream ss;
        {
            cereal::JSONOutputArchive archive(ss);
            archive(scores);
        }

        std::cout << "JSON输出:" << std::endl;
        std::cout << ss.str() << std::endl;

        // 从JSON反序列化
        std::map<std::string, int> loaded_scores;
        {
            cereal::JSONInputArchive archive(ss);
            archive(loaded_scores);
        }

        // 验证数据
        if (scores != loaded_scores)
        {
            std::cerr << "JSON反序列化数据不匹配" << std::endl;
            return false;
        }

        std::cout << "[cereal] JSON序列化测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "JSON序列化测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 测试容器序列化
 * @return 测试是否通过
 */
bool test_container_serialization()
{
    std::cout << "[cereal] 测试容器序列化..." << std::endl;

    try
    {
        // 测试vector序列化
        std::vector<int> original_vec = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

        std::stringstream ss;
        {
            cereal::BinaryOutputArchive archive(ss);
            archive(original_vec);
        }

        std::vector<int> loaded_vec;
        {
            cereal::BinaryInputArchive archive(ss);
            archive(loaded_vec);
        }

        if (original_vec != loaded_vec)
        {
            std::cerr << "Vector序列化失败" << std::endl;
            return false;
        }

        std::cout << "[cereal] 容器序列化测试通过" << std::endl;
        return true;
    }
    catch (const std::exception& e)
    {
        std::cerr << "容器序列化测试失败: " << e.what() << std::endl;
        return false;
    }
}

/**
 * @brief 程序入口
 */
int main()
{
    std::cout << "========================================" << std::endl;
    std::cout << "Cereal 序列化库测试" << std::endl;
    std::cout << "版本: 1.3.2" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    all_passed &= test_binary_serialization();
    all_passed &= test_json_serialization();
    all_passed &= test_container_serialization();

    std::cout << "========================================" << std::endl;
    if (all_passed)
    {
        std::cout << "所有测试通过!" << std::endl;
        return 0;
    }
    else
    {
        std::cout << "部分测试失败!" << std::endl;
        return 1;
    }
}
