
/* Software SPAMS v2.1 - Copyright 2009-2011 Julien Mairal 
 *
 * This file is part of SPAMS.
 *
 * SPAMS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SPAMS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SPAMS.  If not, see <http://www.gnu.org/licenses/>.
 */

/*!
 * \file
 *                toolbox decomp
 *
 *                by Julien Mairal
 *                julien.mairal@inria.fr
 *
 *                File mexL1L2BCD.h
 * \brief mex-file, function mexGroupIST
 * Usage: alpha = mexGroupIST(X,D,alpha0,list_groups,param); 
 * mode, itermax, tol and p are optional
 * mode == 1 correspond to L2ERROR
 * mode == 2 correspond to PENALTY
 * initMean == 0 or 1 default (0)
 * */


#include <mexutils.h>
#include <decomp.h>

template <typename T>
inline void callFunction(mxArray* plhs[], const mxArray*prhs[]) {
   if (!mexCheckType<T>(prhs[0])) 
      mexErrMsgTxt("type of argument 1 is not consistent");
   if (mxIsSparse(prhs[0])) 
      mexErrMsgTxt("argument 1 should be full");
   if (!mexCheckType<T>(prhs[1])) 
      mexErrMsgTxt("type of argument 2 is not consistent");
   if (mxIsSparse(prhs[1])) mexErrMsgTxt("argument 2 should be full");
   if (mxIsSparse(prhs[2])) mexErrMsgTxt("argument 3 should be full");
   if (!mexCheckType<long>(prhs[3])) 
      mexErrMsgTxt("type of argument 4 is not consistent");
   if (!mxIsStruct(prhs[4])) 
      mexErrMsgTxt("argument 5 should be struct");

   T* prX=reinterpret_cast<T*>(mxGetPr(prhs[0]));
   const mwSize* dims=mxGetDimensions(prhs[0]);
   long n=static_cast<long>(dims[0]);
   long M=static_cast<long>(dims[1]);

   T * prD = reinterpret_cast<T*>(mxGetPr(prhs[1]));
   const mwSize* dimsD=mxGetDimensions(prhs[1]);
   long nD=static_cast<long>(dimsD[0]);
   if (nD != n) mexErrMsgTxt("wrong size for argument 2");
   long K=static_cast<long>(dimsD[1]);

   /// modify matrix in place
   const mwSize* dimsA=mxGetDimensions(prhs[2]);
   long Ka = static_cast<long>(dimsA[0]);
   long Ma = static_cast<long>(dimsA[1]);
   if (Ma != M || Ka != K) mexErrMsgTxt("wrong size for argument 3");
   plhs[0]=mxDuplicateArray(prhs[2]);
   T* pr_alpha=reinterpret_cast<T*>(mxGetPr(plhs[0]));

   const mwSize* dimsList = mxGetDimensions(prhs[3]);
   long Ng = static_cast<long>(dimsList[0]*dimsList[1]);
   long* list_groups=reinterpret_cast<long*>(mxGetPr(prhs[3]));

   T lambda= getScalarStruct<T>(prhs[4],"lambda");
   T tol= getScalarStructDef<T>(prhs[4],"tol",1e-3);
   long itermax= getScalarStructDef<long>(prhs[4],"itermax",100);
   long numThreads = getScalarStructDef<long>(prhs[4],"numThreads",-1);
   constraint_type mode= (constraint_type)getScalarStructDef<long>(prhs[4],"mode",PENALTY);


   Matrix<T> D(prD,n,K);

   Matrix<T>* X = new Matrix<T>[Ng];
   Matrix<T>* alpha = new Matrix<T>[Ng];
   if (list_groups[0] != 0)
      mexErrMsgTxt("First group index should be zero");
   for (long i = 0; i<Ng-1; ++i) {
      if (list_groups[i] >= M) 
         mexErrMsgTxt("Size of groups is not consistent");
      if (list_groups[i] >= list_groups[i+1]) 
         mexErrMsgTxt("Group indices should be a strictly non-decreasing sequence");
      X[i].setData(prX+list_groups[i]*n,n,list_groups[i+1]-list_groups[i]);
      alpha[i].setData(pr_alpha+list_groups[i]*K,K,list_groups[i+1]-list_groups[i]);
   }
   X[Ng-1].setData(prX+list_groups[Ng-1]*n,n,M-list_groups[Ng-1]);
   alpha[Ng-1].setData(pr_alpha+list_groups[Ng-1]*K,K,M-list_groups[Ng-1]);

   ist_groupLasso<T>(X,D,alpha,Ng,lambda,mode,itermax,tol,numThreads);

   delete[](X);
   delete[](alpha);
}

void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
   if (nrhs != 5)
      mexErrMsgTxt("Bad number of inputs arguments");

   if (nlhs != 1) 
      mexErrMsgTxt("Bad number of output arguments");

   if (mxGetClassID(prhs[0]) == mxDOUBLE_CLASS) {
      callFunction<double>(plhs,prhs);
   } else {
      callFunction<float>(plhs,prhs);
   }
}

