#ifndef USEFULL_SERIALIZATIONS_H
#define USEFULL_SERIALIZATIONS_H

#include <glm/glm.hpp>
#include "BoundingBox.h"

namespace glm
{
    // Serialization functions are defined in BoundingBox.h
    // ivec3 serialization is kept here
    template<class Archive>
    void serialize(Archive & archive,
                glm::ivec3 & m)
    {
        archive( m.x, m.y, m.z );
    }
}

#endif
