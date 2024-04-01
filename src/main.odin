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

    // rf = math.pow(rf, 1.0/f32(GAMMA));
    // gf = math.pow(gf, 1.0/f32(GAMMA));
    // bf = math.pow(bf, 1.0/f32(GAMMA));


    /*
     . Log of any base bigger than one is trash, we ought to get only negative numbers
     . unless we take abs of the result
    */
    // rf = math.log(rf, 1.0/10);
    // gf = math.log(gf, 1.0/10);
    // bf = math.log(bf, 1.0/10);

    // rf = -math.log(rf, 10);
    // gf = -math.log(gf, 10);
    // bf = -math.log(bf, 10);


    // rf = 1.0 - rf
    // gf = 1.0 - gf
    // bf = 1.0 - bf


    // rf = 20.0 + math.log(f32(r), 10);
    // gf = 20.0 + math.log(f32(g), 10);
    // bf = 20.0 + math.log(f32(b), 10);

    @static min, max :f32 = 0.0, 0.99999
    r = u8(256 * math.clamp(rf, min, max));
    g = u8(256 * math.clamp(gf, min, max));
    b = u8(256 * math.clamp(bf, min, max));

    // mask :u8= 0b0_1_1_1_1_1_1_1
    mask :u8 = 0b1_0_0_0_0_0_0_0
    // mask :u8= 0b0_0_0_0_1_1_1_1
    // r &= mask
    // g &= mask
    // b &= mask

    r = linear_by_parts(r)
    g = linear_by_parts(g)
    b = linear_by_parts(b)


    return {r,b,g,a};
}


linear_by_parts :: proc(
    source: u8,
    s1 : f32 = 89,
    d1 : f32 = 0,
    s2 : f32 = 255,
    d2 : f32 = 255,
) -> u8 {
    d : f32  = 0;
    s : f32  = f32(source)

    MAX : f32 = 255

    if s >= 0 && s < s1 {
        // Segment 1: 0 at s1
        d = (cast(f32)d1 / s1) * s;
    } else if s >= s1 && s <= s2 {
        // Segment 2: s1 at s2
        if s2 - s1 == 0.0 {
            d = d1
        } else {
            d = ((cast(f32)d2 - d1) / (s2 - s1))*(s-s2) +  d2
        }
    } else if s > s2 && s <= 255 {
        // Segment 3: s2 at 255
        if (MAX - d2) == 0.0 {
            d = MAX
        } else {
            d = ((MAX - d2) / (MAX - s2))*(s - MAX) +  MAX
        }
    } else {
        // If s is outside the range [0, 255]
        panic("Valor de s fora do intervalo permitido.\n");
    }
    return u8(d);
}


img_transform ::  #force_inline proc(img: Image) -> Image {
    @static IMG_DATA: [4000*4000]u32;
    data := IMG_DATA[:]

    width, height : i32 = img.width, img.height

    min, max :u8= 255, 0
    for j in 0..<height {
        for i in 0..<width {
            x,y: = i,j
            color := rl.GetImageColor(img,x,y)
            for val in color {
                if val > max do max = val
                if val < min do min = val

            }

            // data[j*width + i] = transmute(u32) color
            data[j*width + i] = transmute(u32) transform(color)
        }
    }
    fmt.printf("min=%v, max=%v\n", min, max)
    // if true do panic("lol")

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
    defer rl.UnloadImage(original_img)
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
