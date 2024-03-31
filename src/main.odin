package pim

main :: proc () {
    rl.TraceLog(.INFO, 
        fmt.ctprintf(
            "Image Height =%v, Width=%v",
            IMAGE_HEIGHT, IMAGE_WIDTH
        )
    )
    assert(SCREEN_HEIGHT*SCREEN_WIDTH > 0)
    assert(IMAGE_HEIGHT*IMAGE_WIDTH > 0)

    loop(render);
}

U32Img :: struct {
    width, height :i32,
    data: [^]u32,
}

Vector3  :: #type rl.Vector3
Color    :: #type rl.Vector3

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

render ::  #force_inline proc(data: []u32) -> U32Img {
        // Get a randomly-sampled camera ray for the pixel at location i,j, originating from
        // the camera defocus disk.

    pixel_sample_square :: proc(pixel_delta_u, pixel_delta_v: Vector3) -> Color {
        // Returns a random point in the square surrounding a pixel at the origin.
        px := -0.5 + rand.float32_range(0, 1.0)
        py := -0.5 + rand.float32_range(0, 1.0)
        return (px * pixel_delta_u) + (py * pixel_delta_v);
    }

    color_u32 :: proc(color: Color, samples_per_pixel: int = 1) -> u32 {
        rf, gf, bf := expand_values(color)
        // Divide the color by the number of samples.
        scale := 1.0 / f32(samples_per_pixel);
        rf *= scale;
        gf *= scale;
        bf *= scale;

        rf = math.pow(rf, 1.0/f32(GAMMA));
        gf = math.pow(gf, 1.0/f32(GAMMA));
        bf = math.pow(bf, 1.0/f32(GAMMA));

        @static min, max :f32 = 0.0, 0.999
        // assert(r <= 1.0 && g <= 1.0 && b <= 1.0)
        r := u32(256 * math.clamp(rf, min, max));
        g := u32(256 * math.clamp(gf, min, max));
        b := u32(256 * math.clamp(bf, min, max));

        a := u32(0xff)
        color_as_u32 :u32 =  (a << 24) | (b << 16) | (g << 8) | (r);
        return color_as_u32;
    }


    camera_lookfrom := Vector3{0,0,0}
    camera_lookat   := Vector3{0,0,-1}
    camera_up       := Vector3{0,1,0}


    // camera_lookfrom = Vector3{-1.5,3, 0.5}
    // camera_lookat   = Vector3{0,0,-2}
    // camera_up       = Vector3{0,1,0}
    width, height : i32 = IMAGE_WIDTH, IMAGE_HEIGHT

    for j in 0..<height {
        for i in 0..<width {
            x,y: f32 = f32(i),f32(j)
            pixel_color := Color{0,0,0}
            data[j*width + i] = color_u32(pixel_color)
        }
    }


    img := U32Img{data = raw_data(data), width = auto_cast width, height = auto_cast height}

    assert(img.height > 1 && img.width > 1)
    return img
}



Render_Func:: #type proc(data: []u32) -> U32Img;
loop :: proc(img_gen :Render_Func) {

    title :: proc () -> cstring {
        return rl.TextFormat("FPS: %v\n", rl.GetFPS())
    }


    rl.InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, title=title());

    rl.SetConfigFlags({.WINDOW_RESIZABLE})



    rl.SetTargetFPS(60);


    @static IMG_DATA: [4000*4000]u32;

    data := IMG_DATA[:]
    img : rl.Image;
    img.format = rl.PixelFormat.UNCOMPRESSED_R8G8B8A8
    img.mipmaps = 1

    u32img : U32Img;

    u32img = img_gen(data)

    img.data = u32img.data;
    img.width = u32img.width;
    img.height = u32img.height;

    texture := rl.LoadTextureFromImage(img);
    { // Render as png just once
        u32img = img_gen(data)
        img.data = u32img.data;
        rl.ExportImage(img, fmt.ctprintf("img.png"))
    }

    for !rl.WindowShouldClose() {
        rl.SetWindowTitle(title())
        rl.BeginDrawing();
        rl.ClearBackground(rl.DARKBLUE);
        u32img = img_gen(data)
        img.data = u32img.data;
        rl.UpdateTexture(texture, img.data);
        rl.DrawFPS(10, 10);
        rl.DrawTexture(texture, SCREEN_WIDTH / 2 - (img.width/2) , SCREEN_HEIGHT / 2 - (img.height/2), rl.WHITE);
        rl.DrawText("This IS a texture loaded from raw image data!", 300, 370, 10, rl.RAYWHITE);
        rl.EndDrawing();
    }

    rl.CloseWindow();
}

import os   "core:os"
import str  "core:strings"
import fmt  "core:fmt"
import mem  "core:mem"
import la   "core:math/linalg"
import math "core:math"
import rand "core:math/rand"

import rl "vendor:raylib"
