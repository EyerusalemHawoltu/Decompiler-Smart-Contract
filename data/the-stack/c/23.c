//
//  strategyM_1.c
//  pstock
//
//  Created by takayoshi on 2016/01/23.
//  Copyright © 2016年 pgostation. All rights reserved.
//

#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <libiomp/omp.h>
#include "main.h"
#include "strMdata.h"
#include "strMsignal.h"
#include "strMbacktest.h"
#include "strLbacktestInit.h"
#include "xmapString.h"

//とりあえずjavaから呼び出して高速にストラテジー計算が行えるようにする

//INPUT:
//  conf.xmapの場所
//  分割補正後の株価xmapの場所
//  _kessandata, _companydataの場所
//  期間、初期金額

//  strategySは使えないとする

//OUTPUT:
//  計算途中の資産額xmap出力(calcxmap) <- 月ごとには出さないことにする
//  allSignals.xmap
//  allShisan.xmap
//  extraSignals.xmap
//  extraResult.xmap
//  allShisan.xmap
//  filter2signal.xmap
//  saveconf.xmap
//  today.xmap

//単純化のため、strategyMを複数の工程に分割する
//<データ構築>
//  銘柄数の分、配列を作る
//  株価データをメモリ上に構築
//  決算データをメモリ上に構築
//  企業データをメモリ上に構築(使うのって株数くらい?)
//<発注シグナル作成>
//  1日ずつ進めながら計算 前後関係ないのでopenmp
//<シミュレーション>
//  1日ずつ進めながら計算
//<ロジック計算>
//  計算部分はどうするか。C言語コンパイル、自前解析、bison/flex

static void kikansitei(xmap_t* xmap, strMdata* data, int* pStartDateIndex, int* pEndDateIndex);

static char* calcPath;

int main(int argc, const char* argv[])
{
    char confPath[512];
    
    getcwd(confPath, sizeof(confPath));
    strcat(confPath, "/conf.xmap");
    xmap_t* xmap = xmap_load(confPath);
    if(xmap==NULL){
        printf("current dir ./conf.xmap is null.\n");
        return 0;
    }
    char* path = xmap_get(xmap, "path");
    calcPath = path;
    
    int isStrL = 0;
    if(strstr(calcPath, "strategyL/")!=NULL) {
        isStrL = 1;
    }

    printf("start %s\n",path);
    printCalc("初期データ取得中", isStrL);
    
    struct timeval startTime, endTime;
    gettimeofday(&startTime, NULL);
    
    char dataPath[256];
    strncpy(dataPath, path, sizeof(dataPath)-1);
    strncat(dataPath, "/../../../", sizeof(dataPath)-1-strlen(dataPath));
    
    strMdata* data = strMdata_initData(dataPath);
    
    strMdata_loadData(dataPath, data);
    
    strMdata_loadKessan(dataPath, data);
    
    //期間指定
    int startDateIndex = 0;
    int endDateIndex = data->dailys[MAX_DATA_COUNT-1].count;
    kikansitei(xmap, data, &startDateIndex, &endDateIndex);
    if (endDateIndex != data->dailys[MAX_DATA_COUNT-1].count) {
        printf("endDateIndex=%d\n",endDateIndex);
        printf("data->dailys[MAX_DATA_COUNT-1].count=%d\n",data->dailys[MAX_DATA_COUNT-1].count);
    }
    
    if(isStrL==0)
    {
        strMsignalSum* signals = strMsignal_start(path, data, startDateIndex, endDateIndex);
        
        printCalc(" strategyM バックテスト中", isStrL);
        
        strMbacktest_start(path, data, signals, startDateIndex, endDateIndex);
        
        gettimeofday(&endTime, NULL);
        if(endTime.tv_usec/1000 - startTime.tv_usec/1000>=0){
            printf("backtest end:%ld.%03d\n", endTime.tv_sec - startTime.tv_sec, endTime.tv_usec/1000 - startTime.tv_usec/1000);
        }else{
            printf("backtest end:%ld.%03d\n", endTime.tv_sec - startTime.tv_sec -1, 1000 + endTime.tv_usec/1000 - startTime.tv_usec/1000);
        }
    }
    else if(isStrL==1)
    {
        strLbacktest_start(path, data, startDateIndex, endDateIndex);
        
        gettimeofday(&endTime, NULL);
        if(endTime.tv_usec/1000 - startTime.tv_usec/1000>=0){
            printf("backtestL end:%ld.%03d\n", endTime.tv_sec - startTime.tv_sec, endTime.tv_usec/1000 - startTime.tv_usec/1000);
        }else{
            printf("backtestL end:%ld.%03d\n", endTime.tv_sec - startTime.tv_sec -1, 1000 + endTime.tv_usec/1000 - startTime.tv_usec/1000);
        }
    }
    
    printCalc(NULL, isStrL);
    
    return 0;
}

void printCalc(char* str, int isStrL)
{
    char dataPath[256];
    strncpy(dataPath, calcPath, sizeof(dataPath)-1);
    strncat(dataPath, "/../../strategyM-calc.xmap", sizeof(dataPath)-1-strlen(dataPath));
    
    xmap_t* xmap = xmap_load(NULL);
    
    if(str!=NULL){
        char buf[512];
        sprintf(buf, "%s %s", calcPath, str);
        xmap_set(xmap, "calcNow", buf);
    } else {
        xmap_set(xmap, "calcNow", "");
    }
    
    xmap_saveWithFilename(xmap, dataPath);
    xmap_release(xmap);
}

static void kikansitei(xmap_t* xmap, strMdata* data, int* pStartDateIndex, int* pEndDateIndex)
{
    int startDateIndex = 0;
    int endDateIndex = data->dailys[MAX_DATA_COUNT-1].count;
    
    char* startDateStr = xmap_get(xmap, "start");
    char* endDateStr = xmap_get(xmap, "end");
    if (startDateStr != NULL)
    {
        printf("startDateStr=%s\n", startDateStr);
        int year=0, month=0, day=0;
        int cnt = 0;
        for (int i=0; i<strlen(startDateStr); i++){
            if (startDateStr[i] == '/') cnt++;
        }
        if (cnt == 1) {
            sscanf(startDateStr, "%d/%d", &year, &month);
            day = 1;
        }
        else if (cnt == 2) {
            sscanf(startDateStr, "%d/%d/%d", &year, &month, &day);
        }
        else {
            printf("startDateStr Error (%s)\n", startDateStr);
        }
        int startDate = ((year-1900)<<16) + (month<<8) + day;
        for (startDateIndex = 0; startDateIndex < data->dailys[MAX_DATA_COUNT-1].count; startDateIndex++) {
            if (data->date[startDateIndex] >= startDate) {
                break;
            }
        }
    }
    if (endDateStr != NULL)
    {
        printf("endDateStr=%s\n", endDateStr);
        int year=0, month=0, day=0;
        int cnt = 0;
        for (int i=0; i<strlen(endDateStr); i++){
            if (endDateStr[i] == '/') cnt++;
        }
        if (cnt == 1) {
            sscanf(endDateStr, "%d/%d", &year, &month);
            day = 31;
        }
        else if (cnt == 2) {
            sscanf(endDateStr, "%d/%d/%d", &year, &month, &day);
        }
        else {
            printf("endDateStr Error (%s)\n", endDateStr);
        }
        int endDate = ((year-1900)<<16) + (month<<8) + day +1;
        for (endDateIndex = 0; endDateIndex < data->dailys[MAX_DATA_COUNT-1].count; endDateIndex++) {
            if (data->date[endDateIndex] >= endDate) {
                break;
            }
        }
    }
    
    (*pStartDateIndex) = startDateIndex;
    (*pEndDateIndex) = endDateIndex;
}