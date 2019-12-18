#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>
#include <vector>
#include "network.h"
#include "utils.h"
#include "parser.h"
#include "option_list.h"
#include "blas.h"
#include "assert.h"
#include "classifier.h"
#include "string.h"
using namespace std;
float max(float *a, int length);
//int min(float *a, int length);
int coordinate(vector< vector<float> > &pointcloud_origin, int lines);
typedef struct ROI{
    int minX;
    int maxX;
    int minY;
    int maxY;
} ROI;

vector< vector<int> > xy;
int main(){
    int c;
    FILE *fp;
    int lines=0;

    fp=fopen("/home/ubuntu/Documents/pointnet2/DFC2019_track4_trainval/Train-Track4/J/JAX_004_PC3.txt", "rb");
    if(fp)
    {
        while((c=fgetc(fp)) != EOF)
            if(c=='\n'){
              lines++;
              //printf("1\n");
            }
        printf("%d\n",lines);

        fclose(fp);
    }
    vector< vector<float> > a(lines, vector<float>(3,0));
    vector< vector<float> > b(lines, vector<float>(3,0));
    vector< vector<float> > pointcloud(lines, vector<float>(2));

    FILE *fpRead=fopen("/home/ubuntu/Documents/pointnet2/DFC2019_track4_trainval/Train-Track4/J/JAX_004_PC3.txt","r");
    if(fpRead)
    {
        for(int i=0;i<lines;i++)
        {
            //fscanf(fpRead, "%s")
            fscanf(fpRead,"%f,%f,%f,%f,%f\n",&a[i][0],&a[i][1],&a[i][2], &pointcloud[i][0], &pointcloud[i][1]);
            pointcloud[i][0] = a[i][0];
            pointcloud[i][1] = a[i][1];
            //printf("%f,%f\n",pointcloud[i][0],pointcloud[i][1]);
            //getchar();
        }
        fclose(fpRead);
    }

    //printf("1\n");
    coordinate(pointcloud,lines);
    //printf("1");
    /*for(int i=0;i<lines;i++){
        printf("x:%d y:%d\n",xy[i][0],xy[i][1]);
    }*/
    image img0 = load_image_color("test_3d_JAX_017.png", 0, 0);  
    for(int i=0;i<lines;i++)
        {b[i][0]=img0.data[0 * 1026 * 1026 + xy[i][0]* 1026 + xy[i][1]] *255.;
	 b[i][1]=img0.data[1 * 1026 * 1026 + xy[i][0]* 1026 + xy[i][1]] *255.;
	 b[i][2]=img0.data[2 * 1026 * 1026 + xy[i][0]* 1026 + xy[i][1]] *255.;
         }
    FILE *fpWrite=fopen("result.txt","w");
   if(fpWrite)  
    {  
        for(int i=0;i<lines;i++)
        {  
        fprintf(fpWrite,"%f,%f,%f,%f,%f,%f\n",a[i][0],a[i][1],a[i][2],b[i][0],b[i][1],b[i][2]); 
        } 
        fclose(fpWrite); 
    }  

    return 0 ;
}//home/ubuntu/Documents/fgpa3d/coordinate.cpp|43|error: cannot convert ‘float (*)[2]’ to ‘float*’ for argument ‘1’ to ‘int coordinate(float*)’|

int coordinate(vector< vector<float> > &pointcloud_origin, int lines){
    float size_cell = 0.5;
    int i,length;
    length = lines;
    //printf("1");
    float pointcloud_x[length];
    float pointcloud_y[length];
    int x[length];
    int y[length];
    vector<int> a(2);
    ROI size_ROI;
    for(i=0;i<length;i++){
        pointcloud_x[i] = pointcloud_origin[i][0];
        pointcloud_y[i] = pointcloud_origin[i][1];
        //printf("%f,%f\n",pointcloud_x[i],pointcloud_y[i]);
    }
    //size_ROI.minX = floor(min(pointcloud_x,length));
    size_ROI.maxX = ceil(max(pointcloud_x,length));
    //size_ROI.minY = floor(min(pointcloud_y,length));
    size_ROI.maxY = ceil(max(pointcloud_y,length));
    //printf("%d %d\n",size_ROI.maxX,size_ROI.maxY);
    for(i=0;i<length;i++){
        pointcloud_x[i] = size_ROI.maxX - pointcloud_x[i];
        x[i] = pointcloud_x[i] / size_cell;
        //printf("%d\n",x[i]);
    }
    for(i=0;i<length;i++){
        pointcloud_y[i] = size_ROI.maxY - pointcloud_y[i];
        y[i] = pointcloud_y[i] / size_cell;
    }
    for(i=0;i<length;i++){
        a[0] = x[i];
        a[1] = y[i];

        xy.push_back(a);
    }
    return 0;
}


float max(float *a, int length){
    float t;
    for(int j=0;j<length;j++){
        //printf("%d %f\n",j,a[j]);
        if(j==0) t=a[0];
        if(t < a[j]) t=a[j];
        }
    //printf("max %f\n",t);
    return t;
}

int min(float *a, int length){
    int t,j;
    for(j=0;j<length;j++){
        if(j==0)
	        t=a[0];
        else if(t > a[j])
            t=a[j];
        }
    return t;
}
