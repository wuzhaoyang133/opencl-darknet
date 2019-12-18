/*
 * sgemm.h
 *
 *  Created on: 2017年12月5日
 *      Author: yiicy
 */

#ifndef OCLINCLUDE_SGEMM_H_
#define OCLINCLUDE_SGEMM_H_

#define USE_VECTOR_LOAD_INSTRUCTION 0
#define TSM 64  //tile size M
#define TSN 64  //tile size N
#define TSK 64  //tile size K
#define PWM  8   //process work M
#define PWN  4  //process work N
#define WNM (TSM / PWM) //workitem number M
#define WNN (TSN / PWN) //workitem number N
#define WLTA ((TSM * TSK) / (WNM * WNN)) //workitem load times A
#define WLTB ((TSK * TSN) / (WNM * WNN)) //workitem load times B
#define PADA 1 //__local memory A pad
#define PADB 1 //__local memory B pad
#define LDW 16  //向量类型长度
#define WVLTA ((TSM * TSK) / (WNM * WNN * LDW)) //workitem vector load times A
#define WVLTB ((TSK * TSN) / (WNM * WNN * LDW)) //workitem vector load times B

#endif /* OCLINCLUDE_SGEMM_H_ */
