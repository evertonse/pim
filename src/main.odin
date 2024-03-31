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

    loop();
}

Image    :: #type rl.Image
Color    :: #type rl.Vector3
Vector3  :: #type rl.Vector3

transform :: proc(color: rl.Color) -> rl.Color {
    r, g, b, a := expand_values(color)
    // if true do return {r,b,g,a};

    rf := f32(r)/f32(255);
    gf := f32(g)/f32(255);
    bf := f32(b)/f32(255);

    rf = math.pow(rf, 1.0/f32(GAMMA));
    gf = math.pow(gf, 1.0/f32(GAMMA));
    bf = math.pow(bf, 1.0/f32(GAMMA));

    // rf := math.log(f32(r), 10);
    // gf := math.log(f32(g), 10);
    // bf := math.log(f32(b), 10);


    // rf = 20.0 + math.log(f32(r), 10);
    // gf = 20.0 + math.log(f32(g), 10);
    // bf = 20.0 + math.log(f32(b), 10);

    @static min, max :f32 = 0.0, 0.999
    r = u8(256 * math.clamp(rf, min, max));
    g = u8(256 * math.clamp(gf, min, max));
    b = u8(256 * math.clamp(bf, min, max));

    return {r,b,g,a};
}

img_transform ::  #force_inline proc(img: Image) -> Image {
    @static IMG_DATA: [4000*4000]u32;
    data := IMG_DATA[:]

    width, height : i32 = img.width, img.height
    for j in 0..<height {
        for i in 0..<width {
            x,y: = i,j
            color := rl.GetImageColor(img,x,y)
            // data[j*width + i] = transmute(u32) color
            data[j*width + i] = transmute(u32) transform(color)
        }
    }

    img := img
    img.data = raw_data(data)
    img.mipmaps = 1
    img.format = rl.PixelFormat.UNCOMPRESSED_R8G8B8A8

    assert(img.height > 1 && img.width > 1)
    return img
}



loop :: proc() {

    title :: proc () -> cstring {
        return rl.TextFormat("FPS: %v\n", rl.GetFPS())
    }


    rl.InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, title=title());
    rl.SetConfigFlags({.WINDOW_RESIZABLE})
    rl.SetTargetFPS(60);



    original_img := rl.LoadImage(IMG_FILE_PATH)
    original_texture := rl.LoadTextureFromImage(original_img);

    img := img_transform(original_img)
    texture := rl.LoadTextureFromImage(img);

    fmt.print(img)
    rl.ExportImage(img, fmt.ctprintf("img.png"))
    // if true do panic("lol")

    for !rl.WindowShouldClose() {
        rl.SetWindowTitle(title())
        rl.BeginDrawing();
        rl.ClearBackground(rl.DARKBLUE);

        img = img_transform(original_img)
        rl.UpdateTexture(texture, img.data);
        rl.DrawFPS(10, 10);

        rl.DrawTexture(original_texture,  0,  SCREEN_HEIGHT / 2 - (img.height/2), rl.WHITE);
        rl.DrawTexture(texture, SCREEN_WIDTH / 2 - (img.width/2) , SCREEN_HEIGHT / 2 - (img.height/2), rl.WHITE);

        font_size : i32 = 20
        rl.DrawText("Original Image", 0,  SCREEN_HEIGHT / 2 - (img.height/2),font_size, rl.BLACK);
        rl.DrawText("Rendered Image", SCREEN_WIDTH / 2 - (img.width/2) , SCREEN_HEIGHT / 2 - (img.height/2), font_size, rl.BLACK);

        
        rl.EndDrawing();
    }
    rl.UnloadImage(img)

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
