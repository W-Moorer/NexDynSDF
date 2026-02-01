#ifndef USEFULL_SERIALIZATIONS_H
#define USEFULL_SERIALIZATIONS_H

#include <glm/glm.hpp>
#include "BoundingBox.h"

namespace glm
{
    // 序列化函数已在 BoundingBox.h 中定义
    // 这里保留 ivec3 的序列化
    template<class Archive>
    void serialize(Archive & archive,
                glm::ivec3 & m)
    {
        archive( m.x, m.y, m.z );
    }
}

#endif
