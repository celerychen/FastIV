/*
 * FastIV - Fast image and vision
 * Copyright (C) 2026 Celery Chen
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 3 as
 * published by the Free Software Foundation.
 *
 * See LICENSE file in project root for full license text.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
typedef float ivf32;
/* 所有 conv2d 入口的 ABI 签名一致（数组参数退化为 ivf32*），可用同一函数指针 */
typedef void (*ConvFn)(ivf32*,int,int,int,ivf32*,int,int,int,ivf32*);
void conv2d_v0(ivf32*,int,int,int,ivf32*,int,int,int,ivf32[9]);
void conv2d_3x1_1x3_v0(ivf32*,int,int,int,ivf32*,int,int,int,ivf32[6]);
void conv2d_3x1_1x3_v1(ivf32*,int,int,int,ivf32*,int,int,int,ivf32[6]);
static double now(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;}
static volatile double sink=0;
static double bench(ConvFn fn,ivf32*d,int wd,int hd,int sd,ivf32*s,int ws,int hs,int ss,ivf32 c[],int it){double t0=now();for(int k=0;k<it;k++){fn(d,wd,hd,sd,s,ws,hs,ss,c);sink+=d[0];}return (now()-t0)/it;}
static double bbest(ConvFn fn,ivf32*d,int wd,int hd,int sd,ivf32*s,int ws,int hs,int ss,ivf32 c[],int it){double b=1e300;for(int t=0;t<5;t++){double p=bench(fn,d,wd,hd,sd,s,ws,hs,ss,c,it);if(p<b)b=p;}return b;}
int main(void){
  /* 可分离核：sobel 类 V=[1,2,1] H=[1,0,-1] */
  ivf32 V[3]={1,2,1}, H[3]={1,0,-1};
  ivf32 ref9[9], sep6[6];
  for(int r=0;r<3;r++)for(int c=0;c<3;c++)ref9[r*3+c]=V[r]*H[c];
  sep6[0]=H[0];sep6[1]=H[1];sep6[2]=H[2];sep6[3]=V[0];sep6[4]=V[1];sep6[5]=V[2];
  int Ns[]={100,500,1000,2000,3000,4000,5000};
  printf("%5s | %9s %9s %9s | %8s %8s %8s | %11s %11s %10s\n","N","v0(ms)","v0sep(ms)","v1(ms)","v0/v0sep","v0/v1","v0sep/v1","v0(Mops)","v1(Mops)","v1 buf");
  for(int z=0;z<6;z++){int N=Ns[z];size_t n=(size_t)N*N;
    ivf32*src=NULL;ivf32*dst=NULL;posix_memalign((void**)&src,64,n*4);posix_memalign((void**)&dst,64,n*4);
    srand(123+N);for(size_t k=0;k<n;k++)src[k]=(ivf32)rand()/RAND_MAX;
    int it=1;double p0=bench(conv2d_v0,dst,N,N,N,src,N,N,N,ref9,it);while(p0*it<0.05&&it<(1<<22)){it*=2;p0=bench(conv2d_v0,dst,N,N,N,src,N,N,N,ref9,it);}
    conv2d_v0(dst,N,N,N,src,N,N,N,ref9);conv2d_3x1_1x3_v0(dst,N,N,N,src,N,N,N,sep6);conv2d_3x1_1x3_v1(dst,N,N,N,src,N,N,N,sep6);
    double b0=bbest(conv2d_v0,dst,N,N,N,src,N,N,N,ref9,it);
    double bS=bbest(conv2d_3x1_1x3_v0,dst,N,N,N,src,N,N,N,sep6,it);
    double b1=bbest(conv2d_3x1_1x3_v1,dst,N,N,N,src,N,N,N,sep6,it);
    printf("%5d | %9.3f %9.3f %9.3f | %7.2fx %7.2fx %7.2fx | %11.1f %11.1f %9.0fB\n",N,b0*1e3,bS*1e3,b1*1e3,b0/bS,b0/b1,bS/b1,(double)n*9/1e6/b0,(double)n*9/1e6/b1,(double)6*N*4);
    free(src);free(dst);
  }
  printf("sink=%f\n",sink);return 0;
}
