package pim

vector3_near_zero :: proc (self: Vector3) -> bool {
    s :f32 = 1e-8;
    return math.abs(self[0]) < s && math.abs(self[1]) < s && math.abs(self[2]) < s;
}

vector3_random :: proc (min :f32 = 0, max:f32 = 1.0) -> Vector3 {
    return {rand.float32_range(min, max), rand.float32_range(min, max), rand.float32_range(min, max)};
}

vector3_random_in_unit_sphere :: proc() -> Vector3 {
    for true {
        p := vector3_random(-1,1)
        if la.dot(p, p) < 1 {
            return p
        }
    }
    return {}
}

vector3_random_unit :: proc() -> Vector3 {
    return la.normalize(vector3_random_in_unit_sphere())
}

vector3_random_on_hemisphere :: proc(normal: Vector3) -> Vector3 {
    on_unit_sphere := vector3_random_unit();
    // In the same hemisphere as the normal
    if la.dot(on_unit_sphere, normal) > 0.0 do return on_unit_sphere;
    return -on_unit_sphere;
}
import os   "core:os"
import str  "core:strings"
import fmt  "core:fmt"
import mem  "core:mem"
import la   "core:math/linalg"
import math "core:math"
import rand "core:math/rand"

import rl "vendor:raylib"

