package pim

Image    :: #type rl.Image
Color    :: #type rl.Vector3
Vector3  :: #type rl.Vector3

Matrix :: struct {
    data :[^]f32,
    height, width: i32
}

image_to_matrix :: proc(img: Image) -> Matrix {
    img := img
    result := Matrix{data = raw_data(make([]f32, img.width * img.height)), height=img.height, width=img.width}

    for y in 0..<img.height {
        for x in 0..<img.width {
            color := rl.GetImageColor(img, x, y)
            intensity := f32(color.r + color.g + color.b) / 3.0
            set(result, x,y, intensity)
        }
    }
    return result
}

matrix_to_image :: proc(
    mat: Matrix,
    color := proc(intensity : f32) -> u32 { 
        c := u8(la.clamp(intensity, 0.0, 255.0))
        return u32(c << 24 | c << 16 | c << 8  | 0xff)
    }
) -> Image {
    img := Image{data = mat.data, height=mat.height, width=mat.width, format=rl.PixelFormat.UNCOMPRESSED_R8G8B8A8, mipmaps=1}
    for y in 0..<mat.height {
        for x in 0..<mat.width {
            intensity := get(mat, x,y) 
            (cast([^]u32)img.data)[y * img.width + x] = color(intensity)
        }
    }
    return img
}

get :: #force_inline proc(mat: Matrix, x,y: i32) -> f32 {
    return mat.data[y * mat.width + x]
}

set :: #force_inline proc(mat: Matrix, x,y: i32, val:f32){
    mat.data[y * mat.width + x] = val
}

median :: proc(image: Matrix, filter_size: i32 = 3) -> Matrix {
    using fmt
    convolved := image
    convolved.data = raw_data(make([]f32, convolved.width*convolved.height))
    for y in 0..<image.height {
        for x in 0..<image.width {
            colors :[dynamic]f32
            fmt.println(x,y)
            for j in 0..<filter_size {
                for i in 0..<filter_size {
                    i,j := i32(i), i32(j)
                    i_offset := i-filter_size/2
                    j_offset := j-filter_size/2
                    color: f32;
                    // Check if the current pixel is within the image bounds
                    if x + i_offset >= 0 && x + i_offset < image.width &&
                       y + j_offset >= 0 && y + j_offset < image.height {
                        color = 0.0
                    } else {
                        color = image.data[y*image.width + x]
                    }
                    append(&colors, color)
                }
            }
            slice.sort(colors[:])

            convolved.data[y*convolved.width + x] = colors[len(colors)/2]
        }
    }
    return convolved
}

convolve :: proc(image: Matrix, kernel: Matrix) -> Matrix {
    using fmt
    assert(kernel.height % 2 == 1)
    assert(kernel.width % 2 == 1)

    convolved := image
    convolved.data = raw_data(make([]f32, convolved.width*convolved.height))
    
    for y in 0..<image.height {
        for x in 0..<image.width {
            sum :f32 = 0
            fmt.println(x,y)
            for j in 0..<kernel.height {
                for i in 0..<kernel.width {
                    i,j := i32(i), i32(j)
                    i_offset := i-kernel.width/2
                    j_offset := j-kernel.height/2
                    img_color: f32;
                    // Check if the current pixel is within the image bounds
                    if x + i_offset >= 0 && x + i_offset < image.width &&
                       y + j_offset >= 0 && y + j_offset < image.height {

                        img_color = image.data[(x + i_offset) + (y + j_offset)*image.width]
                    } else {
                        img_color = 0.0
                    }
                    weight := kernel.data[j*kernel.width + i]
                    sum += img_color * weight
                }
            }
            convolved_color := sum
            convolved.data[x + y*convolved.width] = convolved_color
        }
    }
    return convolved
}


transform :: proc(color: u8) -> u8 {
    c := color
    cf := f32(color)/f32(255);


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
    cu := u8(256 * math.clamp(cf, min, max));

    mask :u8= 0b0_0_0_0_0_1_1_1
    // mask :u8 = 0b1_0_0_0_0_0_0_0
    // mask :u8= 0b0_0_0_0_1_1_1_1

    c &= mask

    // c = linear_by_parts(c)


    return c;
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



kernel_mean_blur :: [3][3]f32{
    {1.0/9, 1.0/9, 1.0/9},
    {1.0/9, 1.0/9, 1.0/9},
    {1.0/9, 1.0/9, 1.0/9},
}

kernel_gaussion_blur_data := [3][3]f32{
    { 1.0/16,  2.0/16,  1.0/16},
    { 2.0/16,  4.0/16,  2.0/16},
    { 1.0/16,  2.0/16,  1.0/16},
}
kernel_gaussion_blur  :=  Matrix{
    data=cast([^]f32)raw_data(kernel_gaussion_blur_data[:]),
    width=3,
    height=3,
}



kernel_laplacian_data := [3][3]f32{
    { 0,  1,  0},
    { 1, -4,  1},
    { 0,  1,  0},
}

kernel_laplacian :=  Matrix{
    data=cast([^]f32)raw_data(kernel_laplacian_data[:]),
    width=3,
    height=3,
}

kernel_laplacian_sharp_data := [3][3]f32{
    { -1,  -1,  -1},
    { -1,   9,  -1},
    { -1,  -1,  -1},
}

kernel_laplacian_sharp :=  Matrix{
    data=cast([^]f32)raw_data(kernel_laplacian_sharp_data[:]),
    width=3,
    height=3,
}

laplacian_sharp :: proc(img: Matrix) -> Matrix {
    edges := convolve(img, kernel_laplacian_sharp)
    return edges 
}

img_transform ::  #force_inline proc(image: Image) -> Image {
    using fmt
    image := image
    rl.ImageColorGrayscale(&image)
    img := image_to_matrix(image)

    kernel_width  :: 3
    kernel_height :: 3


    when !POINT_MODE {
        // img = convolve(img, kernel_gaussion_blur)
        // img = median(img)
        // img = laplacian_sharp(img)
        // img = convolve(img, kernel_laplacian)

        fmt.println("return from img_transform")
        return matrix_to_image(img)
    } 
    


    histogram := [256]f32{}
    for p in 0..<len(histogram) do histogram[p] = 0.0
    for j in 0..<img.height {
        for i in 0..<img.width {
            x,y: = i,j
            intesity := u8(get(img,x,y))
            histogram[intesity] += 1.0
        }
    }
    println(histogram)
    histogram /= f32(img.height * img.width)
    for p in 1..<len(histogram) do histogram[p] += histogram[p-1]
    println(histogram)
    for p in 0..<len(histogram) do histogram[p] *= 256-1
    println(histogram)


    for j in 0..<img.height {
        for i in 0..<img.width {
            x,y: = i,j
            intesity := u8(get(img,x,y))
            new_intensity := f32(histogram[intesity])
            set(img,i, j, new_intensity)
        }
    }

    return matrix_to_image(img)
}

// A-B
minus :: proc(A, B: Matrix) -> Matrix {
    result := deepcopy(A)
    for j in 0..<A.height {
        for i in 0..<A.width {
            intensity_a := A.data[j*A.width + i]
            intensity_b := B.data[j*B.width + i]
            result.data[j*A.width + i] = intensity_a - intensity_b
        }
    }
    return result
}

plus :: proc(A, B: Matrix) -> Matrix {
    result := deepcopy(A)
    for j in 0..<A.height {
        for i in 0..<A.width {
            intensity_a := A.data[j*A.width + i]
            intensity_b := B.data[j*B.width + i]
            result.data[j*A.width + i] = intensity_a + intensity_b
        }
    }
    return result
}


deepcopy :: proc(src: Matrix) -> Matrix {
    result := Matrix{
        data = raw_data(make([]f32, src.width * src.height)),
        width  = src.width,
        height = src.height,
    }
    mem.copy(dst=result.data, src=src.data, len=int(src.width * src.height))
    return result
}


unsharp :: proc(img : Matrix, amount:=1) -> Matrix {
    orig := img
    blur := convolve(orig, kernel_gaussion_blur)
    assert(amount==1,"Not implemented for different amounts")

    res := minus(orig, blur)
    orig = plus(orig,res)
    return orig
}



main :: proc() {
    orig := rl.LoadImage(IMG_FILE_PATH)

    fmt.println("original", orig)

    img  := img_transform(orig)
    fmt.println("transformed", img)

    rl.ExportImage(img, fmt.ctprintf("transformed.png"))
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
import "core:sort"
import "core:slice"
