package pim

ASPECT_RATIO  :f32 = 16.0 / 16.0;

SCREEN_WIDTH  :i32 = 1600;
SCREEN_HEIGHT :i32 = cast(i32) (f32(SCREEN_WIDTH)/ASPECT_RATIO)


IMAGE_WIDTH  : i32 =  SCREEN_WIDTH;
IMAGE_HEIGHT : i32 =  SCREEN_HEIGHT;

// VIEWPORT_WIDTH  :f32: f32(VIEWPORT_HEIGHT* f32(1)/f32(ASPECT_RATIO));
VIEWPORT_WIDTH  :f32 = VIEWPORT_HEIGHT * (f32(IMAGE_WIDTH)/f32(IMAGE_HEIGHT));
VIEWPORT_HEIGHT :f32 = 2.0
GAMMA :: 2

// IMG_FILE_PATH :: "./assets/images/porto.png"
// IMG_FILE_PATH :: "./assets/images/lua.png"
IMG_FILE_PATH :: "./assets/images/salted_1.png"
// IMG_FILE_PATH :: "./assets/images/beans.png"

import math "core:math"


