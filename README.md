# eecs598-project Compression with JPEG

This project implements typical JFIF JPEG compression using a serial C++ implementation and an optimized CUDA GPU implementation. All code is contained in compress.cu, and the final report is available in EECS_598_Report.pdf.


# Background Information

Info about PPM images (used as the raw input image): http://netpbm.sourceforge.net/doc/ppm.html

Process of JPEG compression: https://en.wikipedia.org/wiki/JPEG#JPEG_codec_example

Conversion from RGB to YCbCr: https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion

For JPEG compression 4:2:2 and 4:2:0 chrominance subsampling is common. 
4:2:0 may be used in this project.
Chrominance subsampling: http://dougkerr.net/Pumpkin/articles/Subsampling.pdf

JPEG specification: https://www.w3.org/Graphics/JPEG/itu-t81.pdf
Quantization tables are taken from here (Annex K, p. 143)

Currently the sequential code only works for images of size 8w x 8h where w and h are any positive integers
