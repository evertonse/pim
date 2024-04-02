package pim

Image    :: #type rl.Image
Color    :: #type rl.Vector3
Vector3  :: #type rl.Vector3

Matrix :: struct {
    data :[^]f32,
    height, width: i32
}

convolve :: proc(image: Image, kernel: Matrix) -> Image {
    using fmt
    assert(kernel.height % 2 == 1)
    assert(kernel.width % 2 == 1)
    // Convolution operation
    // Define your convolution kernel here

    // Perform convolution
    convolved := image
    convolved.data = raw_data(make([]u8, 3*convolved.width*convolved.height))
    ColorType::[3]u8
    convolved.format = rl.PixelFormat.UNCOMPRESSED_R8G8B8
    
    for y in 0..<image.height {
        for x in 0..<image.width {
            sum :f32 = 0
            fmt.println(x,y)
            for j in 0..<kernel.height {
                for i in 0..<kernel.width {
                    i,j := i32(i), i32(j)
                    i_offset := i-kernel.width/2
                    j_offset := j-kernel.height/2
                    img_color: rl.Color;
                    // Check if the current pixel is within the image bounds
                    if x + i_offset >= 0 && x + i_offset < image.width &&
                       y + j_offset >= 0 && y + j_offset < image.height {
                        img_color = rl.GetImageColor(image, x + i_offset, y + j_offset)
                    } else {
                        img_color = rl.Color{0, 0, 0, 255}
                    }

                    
                    kernel_val := kernel.data[j*kernel.width + i]

                    weight := kernel_val 
                    if image.format == .UNCOMPRESSED_R8G8B8 {
                        sum += f32(img_color.r+img_color.g + img_color.b + img_color.a)/4.0 * weight
                    } else if image.format == .UNCOMPRESSED_R8G8B8A8 {
                        sum += f32(img_color.r+img_color.g + img_color.b + img_color.a)/4.0 * weight
                    } else {
                        assert(image.format == .UNCOMPRESSED_GRAYSCALE)
                        sum += f32(img_color.r)/1.0 * weight
                    }

                }
            }
            sum = la.clamp(sum, -255, 255)
            // if sum > 255  {
            //     // panic("sum is bigger than 255 that should not happen")
            // }
            // // rl.SetPixelColor(convolved.data, {0,0,0,24}, convolved.format)
            println("sum = ", sum)
            convolved_color := ColorType {
                cast(u8)la.abs(sum),
                cast(u8)la.abs(sum),
                cast(u8)la.abs(sum),
            } if sum > 0 else ColorType {
                0,
                0,
                cast(u8)la.abs(sum),
            } 
            println(convolved_color)
            (cast([^]ColorType)convolved.data)[y*convolved.width + x] = convolved_color
            // rl.SetPixelColor(convolved.data, {200,200,200,200}, convolved.format)
        }
    }
    fmt.println("finished convolving\n")
    return convolved
}


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



kernel_xderivative :: [3][3]f32{
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1},
}

kernel_yderivative :: [3][3]f32{
    { 1,  2,  1},
    { 0,  0,  0},
    {-1, -2, -1},
}


kernel_median_blur :: [3][3]f32{
    {1.0/9, 1.0/9, 1.0/9},
    {1.0/9, 1.0/9, 1.0/9},
    {1.0/9, 1.0/9, 1.0/9},
}

kernel_gaussion_blur :: [3][3]f32{}


img_transform ::  #force_inline proc(img: Image) -> Image {

    kernel_ptr := kernel_median_blur

    kernel_width  :: 3
    kernel_height :: 3
    kernel :=  Matrix{
        data=cast([^]f32)raw_data(kernel_ptr[:]),
        width=kernel_width,
        height=kernel_height,

    }
    if true do return convolve(img, kernel)

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



main :: proc() {
    orig := rl.LoadImage(IMG_FILE_PATH)
    fmt.println("original", orig)

    img  := img_transform(orig)
    fmt.println("transformed", img)

    rl.ExportImage(img, fmt.ctprintf("img.png"))
    fmt.println("Finished Exporting")

}

import os   "core:os"
import str  "core:strings"
import fmt  "core:fmt"
import mem  "core:mem"
import la   "core:math/linalg"
import math "core:math"
import rand "core:math/rand"

import rl "vendor:raylib"
