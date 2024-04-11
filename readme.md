# Run 

    $ python3 src/main.py

# Setup

- Needs ``python3`` installed. Tested with ``python 3.10``.
- Needs `ffmpeg` available in path to generate the video. To install on Ubuntu ``sudo apt install ffmpeg`` or ``sudo snap install ffmpeg``. If it is not found on path, it only generate the images
- Needs ``numpy`` installed, **just for the data structure** as python lists are very slow. Install with ``pip install numpy`` for nice fast matrix. All image processing algorithms are implemented by us.  If `pip` is not install, you can get with ``sudo apt install python3-pip`` on ubuntu.

# Notes

- All generated files goes in the``output`` folder. Except for the main one ``group_x``
