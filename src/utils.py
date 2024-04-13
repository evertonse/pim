import timeit
import functools

try:
    import numpy as np
except:
    print(
        "WARNING: Aparentemente não tem numpy instalado. Usado para estrutura de matriz rápidas. Instale com `pip install numpy`"
    )
    exit(1)


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = timeit.default_timer()
        result = func(*args, **kwargs)
        end_time = timeit.default_timer()
        print(f"INFO: {func.__name__} took {(end_time - start_time):.4f}s to execute.")
        return result

    return wrapper


@timer
def create_video_from_images(images, output_video_path, fps=24):
    try:
        from pathlib import Path

        output_video_path = Path(output_video_path)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
    except:
        print("WARNING: pathlib is not found. Python version might be less than 3.4")

    if len(images) == 0:
        return
    height, width = images[0].shape[:2]

    ffmpeg_cmd = [
        "ffmpeg",  # "-hide_banner", # Hides the output thing
        "-loglevel",
        "error",
        "-y",  # Overwrite file if it already exists
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "rgb24",
        "-r",
        str(fps),  # FPS
        "-i",
        "-",  # Read input from stdin
        "-c:v",
        "libx264",  # IDK
        "-preset",
        "medium",  # IDK, something to do with enconding speed
        "-crf",
        "23",  # lower value means better quality but larger file size
        output_video_path,
    ]
    try:
        import subprocess

        ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
        for img in images:
            if len(img.shape) == 2:
                img = convert_to_rgb(img)
            # Numpy array has tobytes(), but other implementations might also have it
            ffmpeg_process.stdin.write(img.tobytes())
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        return True
    except:
        print(
            "WARNING: Aparentemente `ffmpeg` não está no path desse shell. Instale com  `sudo apt install ffmpeg` se quiser ver o video gerado por este projeto."
        )
        return False

def resize(image, new_height, new_width):
    height, width = image.shape

    # scale factors
    scale_height = height / new_height
    scale_width = width / new_width

    resized_image = np.zeros((new_height, new_width), dtype=image.dtype)

    for y in range(new_height):
        for x in range(new_width):
            src_y = int(y * scale_height)
            src_x = int(x * scale_width)
            resized_image[y, x] = image[src_y, src_x]
    return resized_image

def convert_to_rgb(image):
    assert len(image.shape) == 2
    data = list()
    height = image.shape[0]
    width = image.shape[1]
    for j in range(height):
        for i in range(width):
            data.extend([image[j, i], image[j, i], image[j, i]])
    result = np.array(data, dtype=np.uint8).reshape(height, width, 3)
    return result


def write_ppm_file(filepath, image):
    try:
        from pathlib import Path

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
    except:
        print("WARNING: pathlib is not found. Python version might be less than 3.4")

    height, width = image.shape[:2]
    image = image.astype(int)
    with open(filepath, "w") as f:
        if len(image.shape) == 2:
            image = (invert(image) // 255).astype(int)
            f.write("P1\n")
            f.write("# ExCyber Power Style\n")
            f.write(f"{width} {height}\n")
            for row in image:
                line = "".join(str(pixel) for pixel in row)
                # Important to go no longer than 70 chars
                for i in range(0, len(line), 70):
                    f.write(line[i : i + 70])
                    f.write("\n")
        else:
            f.write("P3\n")
            f.write("# ExCyber Power Style\n")
            f.write(f"{width} {height}\n")
            max_val = "255"
            f.write(f"{max_val}\n")  # Max value
            max_val_len = len(max_val)
            wrote = 0
            for column in image:
                line = "\n".join(
                    [f"{pixel[0]} {pixel[1]} {pixel[2]}" for pixel in column]
                )
                f.write(line)
                f.write("\n")


@timer
def read_ppm_file(filepath):
    """
    https://oceancolor.gsfc.nasa.gov/staff/norman/seawifs_image_cookbook/faux_shuttle/pbm.html
    """
    try:
        with open(filepath, "r") as f:
            header = f.readline().strip()
            if not header == "P1":
                print(
                    f"ERROR: Trying to open a {header} format. But only P1 format is supported."
                )
                exit(1)

            # Skip comment lines
            line = f.readline().strip()
            while line.startswith("#"):
                line = f.readline().strip()

            width, height = map(int, line.strip().split())

            pixel_data = list()
            while line:
                line = (
                    f.readline().replace(" ", "").replace("\n", "")
                )  # white space is ignored
                if line.startswith("#"):
                    continue
                # Each pixel in is represented by a byte containing ASCII '1' or '0', representing black and white respectively. There are no fill bits at the end of a row.
                pixel_data.extend(map(lambda x: 255 if x == "0" else 0, line))
    except FileNotFoundError:
        print(
            f"ERROR: file `{filepath}` does not exist, sorry =P. Try executing the script from the root of the project."
        )
        exit(1)

    array = np.array(pixel_data, dtype=np.uint8).reshape(height, width)
    return array


def invert(image):
    return np.ones(image.shape, dtype=np.uint8) * 255 - image
