# essim

![build](https://github.com/facebookresearch/essim/actions/workflows/cmake.yml/badge.svg)

eSSIM[1] is an evolution of SSIM which improves correlation with
subjective scores by employing Coefficient of Variation (CoV) pooling
and reduces complexity by employing box filters and striding.

This is similar to the original algorithm implemented at
https://github.com/utlive/enhanced_ssim but this project is designed
for use at scale and uses optimized floating point and integer SIMD
instructions.

In addition to the CoV pooled eSSIM score this library will also
calculate the arithmetic mean pooled SSIM score. While CoV pooling has
better correlation to subjective scores, and works well in practice
for evaluating typical compression and scaling artifacts, it can
exhibit unstable behaviors for adversarially generated image
pairs. Therefore, mean pooled SSIM may be needed as a fallback when
evaluated images with unknown distortion types.

[1] A. K. Venkataramanan, C. Wu, A. C. Bovik, I. Katsavounidis and Z. Shahid, "A Hitchhiker’s Guide to Structural Similarity," in IEEE Access, vol. 9, pp. 28872-28896, 2021, doi: 10.1109/ACCESS.2021.3056504
https://ieeexplore.ieee.org/document/9344648

## Building

This project uses CMake to build and depends on `clang-tidy` and
`clang-format` for linting and auto formatting of source files. You
can install both by installing llvm. On OS X you can use homebrew, but
it won't link `llvm` so you need to manualy symlink the related
binaries.

``` shell
brew install llvm
sudo ln -s "$(brew --prefix llvm)/bin/clang-tidy" "/usr/local/bin/clang-tidy"
sudo ln -s "$(brew --prefix llvm)/bin/clang-tidy" "/usr/local/bin/clang-format"
sudo ln -s "$(brew --prefix llvm)/bin/clang-apply-replacements" "/usr/local/bin/clang-apply-replacements"\n
```

To build the project, use the following

``` shell
mkdir build
cd build
cmake ..
make
```

CMake will download and install GoogleTest. To skip building tests you can uses

``` shell
cmake .. -DBUILD_TESTS=OFF
```

Assuming you didn't skip building tests you can run tests using `ctest`

``` shell
ctest --output-on-failure
```

## FFmpeg Integration

You can use ESSIM as an ffmpeg filter, using the included `vf_essim.c`
filter definition in the `ffmpeg` folder, and the associated
patch to ffmpeg to enable building support for `essim`.

## License

ESSIM is licensed the BSD-3 license, see LICENSE file in the root of
this source tree for details.

The ffmpeg filter `vf_essim.c` is licensed under the LGPG license,
same as vf_ssim.c in FFmpeg itself, upon which it is based.
