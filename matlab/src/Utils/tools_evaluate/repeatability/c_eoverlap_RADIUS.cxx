#include <mex.h>
#include <math.h>


void mexFunction(int nlhs,
                 mxArray *plhs[],
                 int nrhs,
                 const mxArray *prhs[])
{
    // must have single double array input.
    if (nrhs != 4) { printf("in nrhs != 4\n"); return; }
    if (mxGetClassID(prhs[0]) != mxDOUBLE_CLASS) { printf("input must be a double array\n"); return; }
    if (mxGetClassID(prhs[1]) != mxDOUBLE_CLASS) { printf("input must be a double array\n"); return; }

    // must have single output.
    if (nlhs != 4){ printf("out nlhs != 4\n"); return; }

    // get size of x.
    int const num_dims1 = mxGetNumberOfDimensions(prhs[0]);
    int const *dims1 = mxGetDimensions(prhs[0]);
    int const num_dims2 = mxGetNumberOfDimensions(prhs[1]);
    int const *dims2 = mxGetDimensions(prhs[1]);
    int const num_dims3 = mxGetNumberOfDimensions(prhs[2]);
    int const *dims3 = mxGetDimensions(prhs[2]);
    int const num_dims4 = mxGetNumberOfDimensions(prhs[3]);
    int const *dims4 = mxGetDimensions(prhs[3]);
    if(dims1[0]<9 || dims2[0]<9 || dims1[0]!=dims2[0] || dims3[0]!=1 || dims4[0]!=1){ printf("dims != 9\n"); return; }

    // create new array of the same size as x.

    int *odim = new int[2];
    odim[0]=dims2[1];
    odim[1]=dims1[1];
    plhs[0] = mxCreateNumericArray(2, odim, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxCreateNumericArray(2, odim, mxDOUBLE_CLASS, mxREAL);
    plhs[2] = mxCreateNumericArray(2, odim, mxDOUBLE_CLASS, mxREAL);
    plhs[3] = mxCreateNumericArray(2, odim, mxDOUBLE_CLASS, mxREAL);

    // get pointers to beginning of x and y.
    double const *feat1 = (double *)mxGetData(prhs[0]);
    double const *feat2 = (double *)mxGetData(prhs[1]);
    double const *flag = (double *)mxGetData(prhs[2]);
    double       *over_out  = (double *)mxGetData(plhs[0]);
    double       *mover_out = (double *)mxGetData(plhs[1]);
    double       *desc_out  = (double *)mxGetData(plhs[2]);
    double       *mdesc_out = (double *)mxGetData(plhs[3]);

    float *feat1a = new float[9];
    float *feat2a = new float[9];
    float *tdesc_out = new float[dims2[1]*dims1[1]];
    float *tover_out = new float[dims2[1]*dims1[1]];

    int common_part = (int)flag[0];
    double *p_radius = (double *)mxGetData(prhs[3]);
    float radius = (float)p_radius[0];
    radius *= radius;

    for(int j = 0; j < dims1[1]; j++){ 
        for (int i = 0; i < dims2[1]; i++){
            over_out[j*dims2[1]+i] = 100.0;
            desc_out[j*dims2[1]+i] = 1000000.0;
        }
    }

   for(int j = 0,f1 = 0; j < dims1[1]; j++, f1 += dims1[0]){

       for (int i = 0, f2 = 0; i < dims2[1]; i++, f2 += dims1[0]){
           
           //compute shift error between ellipses
           float dx = feat2[f2] - feat1[f1];
           float dy = feat2[f2+1] - feat1[f1+1];
           float dist = dx*dx+dy*dy;
           if(dist < radius){
               tover_out[j*dims2[1]+i] = 0.0;
               mover_out[j*dims2[1]+i] = 0.0;
           }else{
               tover_out[j*dims2[1]+i] = 100.0;
               mover_out[j*dims2[1]+i] = 100.0;
           }
           float descd = 0;
           for(int v = 9;v < dims1[0]; v++){
               descd += ((feat1[f1+v]-feat2[f2+v])*(feat1[f1+v]-feat2[f2+v]));
           }
           descd = sqrt(descd);
           tdesc_out[j*dims2[1]+i] = descd;
           mdesc_out[j*dims2[1]+i] = descd;
       }
   }

  float minr = 100;
  int mini = 0;
  int minj = 0;
  do{
      minr=100;
      for(int j=0;j<dims1[1];j++){    
          for (int i=0; i<dims2[1]; i++){
              if(minr>tover_out[j*dims2[1]+i]){
                  minr=tover_out[j*dims2[1]+i];
                  mini=i;
                  minj=j;
              }
          }
      }
      if(minr<100){
          for(int j=0;j<dims1[1];j++){
              tover_out[j*dims2[1]+mini]=100;
          }   
          for (int i=0; i<dims2[1]; i++){
              tover_out[minj*dims2[1]+i]=100;
          }
          over_out[minj*dims2[1]+mini]=minr;
      }
  }while(minr<70);


  int dnbr=0;
  do{
      minr=1000000;
      for(int j=0;j<dims1[1];j++){    
          for (int i=0; i<dims2[1]; i++){
              if(minr>tdesc_out[j*dims2[1]+i]){
                  minr=tdesc_out[j*dims2[1]+i];
                  mini=i;
                  minj=j;
              }
          }
      }
      if(minr<1000000){
          for(int j=0;j<dims1[1];j++){
              tdesc_out[j*dims2[1]+mini]=1000000;
          }   
          for (int i=0; i<dims2[1]; i++){
              tdesc_out[minj*dims2[1]+i]=1000000;
          }
          desc_out[minj*dims2[1]+mini]=dnbr++;//minr
      }
  }while(minr<1000000);



   delete []odim;
   delete []tdesc_out;
   delete []tover_out;
   delete []feat1a;
   delete []feat2a;
}
