// Copyright 2006, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#include <cstdio>
#include "gtest/gtest.h"
#define FR_LVL_eSSIM_SUPPORT 0
#define GET_MAX_ABS_INTvsFLOAT_FRAME 1
#if FR_LVL_eSSIM_SUPPORT
#include <string>
#include <iostream>
#include <fstream>
#include <cmath>
#include "essim.h"
#endif

#if GTEST_OS_ESP8266 || GTEST_OS_ESP32
#if GTEST_OS_ESP8266
extern "C" {
#endif
void setup() {
  testing::InitGoogleTest();
}

void loop() { RUN_ALL_TESTS(); }

#if GTEST_OS_ESP8266
}
#endif

#else

#if !FR_LVL_eSSIM_SUPPORT
GTEST_API_ int main(int argc, char **argv) {
  printf("Running main() from %s\n", __FILE__);
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
#endif

#if FR_LVL_eSSIM_SUPPORT
void print_help() {
  printf("\n");
  printf(" -r : Reference yuv file path \n");
  printf(" -d : Distorted yuv file path \n");
  printf(" -w : Width of yuv. Max Width <= 7680, \
                to avoid huge precision loss in eSSIM score \n");
  printf(" -h : Height of yuv. Max Height <= 4320, \
                to avoid huge precision loss in eSSIM score \n");
  printf(" -bd : Bit-depth. Default Bit-depth is 8 \n");
  printf(" -wsize : Window size for eSSIM (can be 8 or 16).Default is 16. \n");
  printf(" -wstride : Window stride for eSSIM (can be 4, 8, 16). \
            Default, consider Window size as window stride. wstride <= wsize \n");
  /*Currently not supporting essimmode as user argument,
  because at frame level we need both INT and Float eSSIM values. */          
  /*printf(" -mode : Can be 0 -> SSIM_MODE_REF, 1 -> SSIM_MODE_PERF_INT, 
  2 -> SSIM_MODE_PERF_FLOAT. Default is 1 \n"); */
  printf("\n Example cmd : \t");
  printf(" -r /mnt/d/InpYuvPath/xyz.yuv -d /mnt/d/ReconYuvPath/abc.yuv \
        -w 1280 -h 720 -bd 10 -wsize 16 -wstride 8 \n");
  printf("\n");
}
uint32_t GetTotalWindowsInFrame(uint32_t FrWd, uint32_t FrHt,
         uint32_t WSize, uint32_t Wstride) {
  uint32_t TotalWinds=0;
  TotalWinds = (1+((FrWd - WSize)/Wstride)) * (1+((FrHt - WSize)/Wstride));
  return TotalWinds;
}

int main(int argc, char **argv) {
  uint32_t i = 1;
  std::string InpYuvPath = "NULL", ReconYuvPath = "NULL";
  uint32_t Width = 0, Height = 0, WSize = 8, WStride = 8, Mode = 1, BitDepth = 8;
  eSSIMMode ModeEnum;

  /*Read cmd line args*/
  if(argc <=2) {
    print_help();
    return 0;
  }
  else {
    while((i < (uint32_t)argc) && (strcmp(argv[i], "") !=0 )) {
      if(strcmp(argv[i],"-r")==0) {
        InpYuvPath = argv[++i];
        std::cout << "RefYuvPath :" << InpYuvPath << std::endl;
      }
      else if (strcmp(argv[i], "-d")==0) {
        ReconYuvPath = argv[++i];
        std::cout << "DistortedYuvPath :" << ReconYuvPath << std::endl;
      }
      else if (strcmp(argv[i], "-w")==0) {
        Width = (uint32_t)atoi(argv[++i]);
        std::cout << "Width :" << Width << std::endl;
      }
      else if (strcmp(argv[i],"-h")==0) {
        Height = (uint32_t)atoi(argv[++i]);
        std::cout << "Height :" << Height << std::endl;
      }
      else if (strcmp(argv[i],"-wsize")==0) {
        WSize = (uint32_t)atoi(argv[++i]);
        std::cout << "WSize :" << WSize << std::endl;
      }
      else if (strcmp(argv[i],"-wstride")==0) {
        WStride = atoi(argv[++i]);
        std::cout << "WStride :" << WStride << std::endl;
      }
      else if (strcmp(argv[i],"-mode")==0) {
        Mode = atoi(argv[++i]);
        if(Mode != 0 && Mode != 1 && Mode != 2) {
          Mode = 1;
          std::cout << "Considering default eSSIMMode i.e 1 (SSIM_MODE_PERF_INT)"
                    << std::endl;
        }
      }
      else if (strcmp(argv[i],"-bd")==0) {
        BitDepth = atoi(argv[++i]);
        std::cout << "BitDepth :" << BitDepth << std::endl;
      }      
      else {
        std::cout << "Unknow argument :" << argv[i] << std::endl;
        return 0;
      }
      i++;
    } 
    if((WSize !=8) && (WSize !=16)) {
      WSize = 16;
      std::cout << "Considering default WSize i.e 16" << std::endl;
    }
    if((WStride > WSize) || ((WStride !=4) && (WStride != 8) && (WStride !=16))) {
      WStride = WSize;
      std::cout << "Considering default WStride as WSize" << std::endl;
    }
  
  }

  if(Mode == 0)
    ModeEnum = SSIM_MODE_REF;
  else if (Mode == 2)
    ModeEnum = SSIM_MODE_PERF_FLOAT;
  else
    ModeEnum = SSIM_MODE_PERF_INT;

  if(ModeEnum == SSIM_MODE_REF) {
    std::cout << "currently not supporting at frame level" << std::endl;
    return 0;
  }

  uint32_t BitDepthMinus8 = BitDepth - 8;

  std::fstream InpYuvfp, ReconYuvfp;

  InpYuvfp.open(InpYuvPath, std::ios::in | std::ios::binary);
  ReconYuvfp.open(ReconYuvPath, std::ios::in | std::ios::binary);

  if(!InpYuvfp) {
    std::cout << "Input Yuv file is empty (or)" << 
              "doesn't found in the specified path" << std::endl;
    InpYuvfp.close();
    return 0;
  }
  if(!ReconYuvfp) {
    std::cout << "Recon Yuv file is empty (or)" <<
              "doesn't found in the specified path" << std::endl;
    ReconYuvfp.close();
    InpYuvfp.close();
    return 0;
  }

  /*Get yuv name from recon path, so we can generate csv file,
    based on that name*/
  char *ptrYuvName;
  char temp[ReconYuvPath.length() +1];
  std::string YuvFileName;
  strcpy(temp,ReconYuvPath.c_str());
  ptrYuvName = strtok(temp, " /");
  while (ptrYuvName != NULL) {
    //std::cout << ptrYuvName << std::endl;
    YuvFileName = ptrYuvName;
    ptrYuvName = strtok(NULL, "/");
  }
  YuvFileName.resize(YuvFileName.size() - 4);
  std::string x_str = "x";
  std::string _str = "_";
  std::string ESSIMcsvFileName = "./ESSIM_WsizeXWstride_" + std::to_string(WSize)+ x_str
                                  + std::to_string(WStride) + _str + YuvFileName + ".csv";
  std::cout << "ESSIM csv FileName : " << ESSIMcsvFileName << std::endl;

  /*Creating o/p file pointers*/
  std::ofstream ESSIMcsvFile(ESSIMcsvFileName);

  /*Writing to the file*/
  ESSIMcsvFile << "FrameNumber, width, height, windSize, windStride,"
               << "IntSSIMscore, FloatSSIMscore, ABSIntFloatSSIMscore,"
               << "IntESSIMscore, FloatESSIMscore, ABSIntFloatESSIMscore,"
               << "InpYuv, ReconYuv" << std::endl;
  
  /*Calculating total num of frames and Frame, Y-Plane & UV-Plnae sizes*/
  InpYuvfp.seekg(0, std::ios::end);
  uint64_t InpYuvFileSize = InpYuvfp.tellg();
  InpYuvfp.seekg(std::ios::beg);

  ReconYuvfp.seekg(0, std::ios::end);
  uint64_t ReconYuvFileSize = ReconYuvfp.tellg();
  ReconYuvfp.seekg(std::ios::beg);

  uint64_t TotalNumOfFrames = 0, FileSize= 0, FrNum=0;
  uint64_t FrSize = 0, YPlaneSize=0, UVPlaneSize = 0;

  /*For 8 bit-depth DataTypeSize = sizeof(uint8_t)=1*/
  uint32_t DataTypeSize = 1;
  if(BitDepth > 8) {
    /*For > 8 bit-depth DataTypeSize = sizeof(uint16_t)=2*/
    DataTypeSize = 2;
  }

  FileSize = std::min(InpYuvFileSize,ReconYuvFileSize);
  if(InpYuvFileSize != ReconYuvFileSize ) {
    std::cout << "Inp yuv & Recon yuv file sizes are not same" << std::endl;
  }

  TotalNumOfFrames = (FileSize * 2)/(DataTypeSize * Width * Height * 3);
  FrSize = (DataTypeSize * Width * Height * 3) / 2;
  YPlaneSize = DataTypeSize * Width * Height;
  UVPlaneSize = FrSize - YPlaneSize;

#if DEBUG_PRINTS  
  printf("\n Fr Width : %d, Fr Height : %d", Width, Height);
  printf("\n Wind Size : %d, Wind Stride : %d", WSize, WStride);
#endif
  uint8_t *InpYuvBuff = NULL, *ReconYuvBuff = NULL;
  uint16_t *InpYuvBuffHbd = NULL, *ReconYuvBuffHbd = NULL;
  ptrdiff_t stride = (sizeof(uint8_t) * Width) & -6 /*LOG2_ALIGN*/;
  if(BitDepth == 8) {
    InpYuvBuff = new uint8_t[YPlaneSize];
    ReconYuvBuff = new uint8_t[YPlaneSize];
  } else {
    InpYuvBuffHbd = new uint16_t[YPlaneSize];
    ReconYuvBuffHbd = new uint16_t[YPlaneSize];
  }

  float FrSSIMScore_float, FrESSIMScore_float, FrSSIMScore_Int, FrESSIMScore_Int;
  float ABSIntFloatSSIMscore, ABSIntFloatESSIMscore;
#if GET_MAX_ABS_INTvsFLOAT_FRAME
  float MAXABSIntFloatESSIMscore = -10.0;
  float MAXABSIntFloatSSIMscore = 0.0;
  float MAXABSFrSSIMScore_float = 0.0, MAXABSFrESSIMScore_float = 0.0;
  float MAXABSFrSSIMScore_Int = 0.0, MAXABSFrESSIMScore_Int = 0.0;
  int64_t MAXABSESSSIMFrNum =0;
#endif
#if PROFILING_PRINTS
  uint32_t numWindows = GetTotalWindowsInFrame(Width, Height, WSize, WStride);
  printf("\t TotalnumWindows in a frame: %i \n",numWindows);
  clock_t start=0, end=0;
  double cpu_time_used=0;
#endif
  for(FrNum = 0; FrNum < TotalNumOfFrames; FrNum++) {
    FrSSIMScore_float = 0.0;
    FrESSIMScore_float = 0.0;
    FrSSIMScore_Int = 0.0;
    FrESSIMScore_Int = 0.0;

    if(BitDepth == 8) {
      memset(InpYuvBuff, 0, YPlaneSize);
      memset(ReconYuvBuff, 0, YPlaneSize);

      InpYuvfp.read((char*)InpYuvBuff, YPlaneSize);
      ReconYuvfp.read((char*)ReconYuvBuff, YPlaneSize);
#if PROFILING_PRINTS
        start=0, end=0;
        cpu_time_used=0;
        start = clock();
#endif
      ssim_compute_8u(&FrSSIMScore_Int, &FrESSIMScore_Int, InpYuvBuff, stride,
                                      ReconYuvBuff, stride, Width, Height, WSize,
                                      WStride, 1, SSIM_MODE_PERF_INT,
                                      SSIM_SPATIAL_POOLING_BOTH);
#if PROFILING_PRINTS   
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("\t cpu_time_used_Int: %lf microsecs\n",cpu_time_used*1000000);
        start=0, end=0;
        cpu_time_used=0;
        start = clock();
#endif
      ssim_compute_8u(&FrSSIMScore_float, &FrESSIMScore_float, InpYuvBuff, stride,
                                      ReconYuvBuff, stride, Width, Height, WSize,
                                      WStride, 1, SSIM_MODE_PERF_FLOAT,
                                      SSIM_SPATIAL_POOLING_BOTH);
#if PROFILING_PRINTS   
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("\t cpu_time_used_Float: %lf microsecs\n",cpu_time_used*1000000);
#endif                                      
    }else {
      memset(InpYuvBuffHbd, 0, YPlaneSize);
      memset(ReconYuvBuffHbd, 0, YPlaneSize);

      InpYuvfp.read((char*)InpYuvBuffHbd, YPlaneSize);
      ReconYuvfp.read((char*)ReconYuvBuffHbd, YPlaneSize);
#if PROFILING_PRINTS
        start=0, end=0;
        cpu_time_used=0;
        start = clock();
#endif
      ssim_compute_16u(&FrSSIMScore_Int, &FrESSIMScore_Int, InpYuvBuffHbd, stride,
                                      ReconYuvBuffHbd, stride, Width, Height, BitDepthMinus8,
                                      WSize, WStride, 1, SSIM_MODE_PERF_INT,
                                      SSIM_SPATIAL_POOLING_BOTH);
#if PROFILING_PRINTS   
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("\t cpu_time_used_Int: %lf microsecs\n",cpu_time_used*1000000);
        start=0, end=0;
        cpu_time_used=0;
        start = clock();
#endif
      ssim_compute_16u(&FrSSIMScore_float, &FrESSIMScore_float, InpYuvBuffHbd, stride,
                                      ReconYuvBuffHbd, stride, Width, Height, BitDepthMinus8,
                                      WSize, WStride, 1, SSIM_MODE_PERF_FLOAT,
                                      SSIM_SPATIAL_POOLING_BOTH);
#if PROFILING_PRINTS   
        end = clock();
        cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        printf("\t cpu_time_used_Float: %lf microsecs\n",cpu_time_used*1000000);
#endif
    }                                
#if DEBUG_PRINTS
    printf("\n FrNum: %ld, SSIMInt : %f, SSIMFlt : %f, ESSIMInt : %f, ESSIMFlt : %f",\
            FrNum, FrSSIMScore_Int, FrSSIMScore_float,FrESSIMScore_Int,FrESSIMScore_float);
#endif
    ABSIntFloatSSIMscore = fabs(FrSSIMScore_Int - FrSSIMScore_float);
    ABSIntFloatESSIMscore= fabs(FrESSIMScore_Int - FrESSIMScore_float);

    InpYuvfp.seekg(UVPlaneSize, std::ios::cur);
    ReconYuvfp.seekg(UVPlaneSize, std::ios::cur);

    /*Writing to a csv file*/
    ESSIMcsvFile << FrNum <<"," << Width <<"," << Height << "," << WSize << ","
                 << WStride << ",";
    ESSIMcsvFile << FrSSIMScore_Int << "," << FrSSIMScore_float << ","
                 << ABSIntFloatSSIMscore << ",";
    ESSIMcsvFile << FrESSIMScore_Int << "," << FrESSIMScore_float << ","
                 << ABSIntFloatESSIMscore << ",";
    ESSIMcsvFile << InpYuvPath << "," << ReconYuvPath << std::endl;
#if GET_MAX_ABS_INTvsFLOAT_FRAME
    if(MAXABSIntFloatESSIMscore < ABSIntFloatESSIMscore) {
      MAXABSIntFloatESSIMscore = ABSIntFloatESSIMscore;
      MAXABSIntFloatSSIMscore = ABSIntFloatSSIMscore;
      MAXABSESSSIMFrNum = FrNum;
      MAXABSFrSSIMScore_float = FrSSIMScore_float;
      MAXABSFrESSIMScore_float = FrESSIMScore_float;
      MAXABSFrSSIMScore_Int = FrSSIMScore_Int;
      MAXABSFrESSIMScore_Int = FrESSIMScore_Int;
    }
#endif 
  }
  printf("\n");
#if  GET_MAX_ABS_INTvsFLOAT_FRAME
    ESSIMcsvFile << "MAX_ESSIM_ABS_FRAME" << std::endl;
    ESSIMcsvFile << MAXABSESSSIMFrNum <<"," << Width <<"," << Height << ","
                 << WSize << "," << WStride << ",";
    ESSIMcsvFile << MAXABSFrSSIMScore_Int << "," << MAXABSFrSSIMScore_float << ","
                 << MAXABSIntFloatSSIMscore << ",";
    ESSIMcsvFile << MAXABSFrESSIMScore_Int << "," << MAXABSFrESSIMScore_float << ","
                 << MAXABSIntFloatESSIMscore << ",";
    ESSIMcsvFile << InpYuvPath << "," << ReconYuvPath << std::endl;
    /*console print*/
    std::cout << "MAX_ESSIM_ABS_FRAME," <<  MAXABSESSSIMFrNum << "," << Width <<","
              << Height << "," << WSize << "," << WStride << ",";
    std::cout << MAXABSFrSSIMScore_Int << "," << MAXABSFrSSIMScore_float << ","
              << MAXABSIntFloatSSIMscore << ",";
    std::cout << MAXABSFrESSIMScore_Int << "," << MAXABSFrESSIMScore_float << ","
              << MAXABSIntFloatESSIMscore << ",";
    std::cout << InpYuvPath << "," << ReconYuvPath << std::endl;
#endif

  if(BitDepth == 8) {
    delete[] InpYuvBuff;
    delete[] ReconYuvBuff;
  } else {
    delete[] InpYuvBuffHbd;
    delete[] ReconYuvBuffHbd;
  }
  InpYuvfp.close();
  ReconYuvfp.close();
  ESSIMcsvFile.close();
  return 0;
}
#endif
#endif
