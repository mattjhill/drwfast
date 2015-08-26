#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "tridiagonal.h"
#include <math.h>
#include <stdlib.h>

void trimult(double *a, double *b, double *c, double *x, double *y, int n) {
	int n1 = n-1;

    y[0] = b[0]*x[0] + c[0]*x[1];
    for (int i=1; i < n1; i++) {
        y[i] = a[i-1]*x[i-1] + b[i]*x[i] + c[i]*x[i+1];
    }
    y[n1] = a[n1-1]*x[n1-1] + b[n1]*x[n1];
}

void trisolve(double *a, double *b, double *c, double *x, double *y, int n) {

    int n1 = n-1;
    int n2 = n1-1;
    double bb, d[n], e[n];

    d[0] = -c[0]/b[0];
    e[0] = y[0]/b[0];

    for (int i=1; i < n1; i++) {
    	bb = b[i] + a[i-1]*d[i-1];
    	d[i] = -c[i]/bb;
    	e[i] = (y[i]-a[i-1]*e[i-1])/bb;
    }
    x[n1] = (y[n1]-a[n2]*e[n2])/(b[n1]+a[n2]*d[n2]);

    for (int k=n2; k > -1; k--) {
    	x[k] = e[k] + d[k]*x[k+1];
    }
}

double tridet(double *a, double *b, double *c, int n) {
	int n1 = n-1;
	int n2 = n1-1;
	double logdet;
	double dd, bb;
    double signdet; 

    signdet = b[0]/fabs(b[0]);
    logdet = log(fabs(b[0]));
    dd=-c[0]/b[0];
    for (int i=1; i < n1; i++) {
        bb = b[i]+a[i-1]*dd;
        dd = -c[i]/bb;
        signdet = signdet*bb/fabs(bb);
        logdet += log(fabs(bb));
    }
    bb = b[n1]+a[n2]*dd;
    signdet=signdet*bb/fabs(bb);
    logdet += log(fabs(bb));
    return logdet;
}

void snsolve(double *t, double *err2, double var, double tcorr, double *x, double *y, int n) {
	int n1, n2;
	n1 = n-1;
	n2 = n1-1;
	double r[n1], e[n1], z[n];
    double a[n1], b[n], c[n1];
    double aa[n1], bb[n], cc[n1];
    double varinv;

    if (n == 1) {
    	x[0] = y[0]/(var+err2[0]);
    }

	varinv = 1./var;

	for (int i=0; i < n1; i++) {
        r[i]=exp(-(t[i+1]-t[i])/tcorr);
        e[i]=r[i]/(1.-pow(r[i],2.0));
	}


	b[0] =  varinv*(1+r[0]*e[0]);
    c[0] = -varinv*e[0];
    for(int i=1; i < n1; i++) {
        a[i-1] = -varinv*e[i-1];
        b[i] = varinv*(1. + r[i]*e[i]+r[i-1]*e[i-1]);
        c[i] = -varinv*e[i];
    }
    a[n2] = -varinv*e[n2];
    b[n1] = varinv*(1.+r[n2]*e[n2]);

    bb[0] = 1. + b[0]*err2[0];
    cc[0] = c[0]*err2[0];
    for(int i=1; i < n1; i++) {
        aa[i-1] = a[i-1]*err2[i];
        bb[i] = 1. + b[i]*err2[i];
        cc[i] = c[i]*err2[i];
    }
    aa[n2] = a[n2]*err2[n1];
    bb[n1] = 1. + b[n1]*err2[n1];

    trisolve(aa,bb,cc,z,y,n);
    trimult(a,b,c,z,x,n); 
}

double snsolve_retdet(double *t, double *err2, double var, double tcorr, double *x, double *y, int n) {
	int n1, n2;
	n1 = n-1;
	n2 = n1-1;
	double r[n1], e[n1], z[n], ldetmat, ldets, ldetm, ldetc;
    double a[n1], b[n], c[n1];
    double aa[n1], bb[n], cc[n1];
    double varinv;

    if (n == 1) {
    	x[0] = y[0]/(var+err2[0]);
    	ldetc = log(var+err2[0]);
    	return ldetc;
    }

	varinv = 1./var;
	ldets = log(var);

	for (int i=0; i < n1; i++) {
        r[i]=exp(-(t[i+1]-t[i])/tcorr);
        e[i]=r[i]/(1.-pow(r[i],2.0));
        ldets=ldets+log(var*(1.-pow(r[i],2.0)));
	}


	b[0] =  varinv*(1+r[0]*e[0]);
    c[0] = -varinv*e[0];
    for(int i=1; i < n1; i++) {
        a[i-1] = -varinv*e[i-1];
        b[i] = varinv*(1. + r[i]*e[i]+r[i-1]*e[i-1]);
        c[i] = -varinv*e[i];
    }
    a[n2] = -varinv*e[n2];
    b[n1] = varinv*(1.+r[n2]*e[n2]);

    bb[0] = 1. + b[0]*err2[0];
    cc[0] = c[0]*err2[0];
    for(int i=1; i < n1; i++) {
        aa[i-1] = a[i-1]*err2[i];
        bb[i] = 1. + b[i]*err2[i];
        cc[i] = c[i]*err2[i];
    }
    aa[n2] = a[n2]*err2[n1];
    bb[n1] = 1. + b[n1]*err2[n1];

    ldetm = tridet(aa,bb,cc,n);
    ldetmat=ldets+ldetm;

    trisolve(aa,bb,cc,z,y,n);
    trimult(a,b,c,z,x,n);
    return ldetmat;  

}

double lnlike(double var, double tcorr, double *t, double *y, double *err2, int n) {
    double L[n];
    double a[n];
    double b[n];
    double c_arr[n];
    double d[n];
    double e[n];
    double ldetc;

    fill(L, 1.0, n);
    /* Get the determinant of C^-1 and solve C a = L */
    ldetc = snsolve_retdet(t, err2, var, tcorr, a, L, n);
    /* Calculate Cq = 1/(L^T C^-1 L) */
    double Cq = 1/dot(L, a, n);
    /* solve C b = y */
    snsolve(t, err2, var, tcorr, b, y, n);
    double c = dot(L, b, n);
    c *= Cq;

    fill(c_arr, c, n);
    snsolve(t, err2, var, tcorr, d, c_arr, n);
    ewise_subtract(e, b, d, n);
    double chisq = dot(y, e, n);
    double lnl = -0.5*chisq + 0.5*log(Cq) - 0.5*ldetc;

    return lnl;
}

double chisq(double var, double tcorr, double *t, double *y, double *err2, int n) {
    double L[n];
    double a[n];
    double b[n];
    double c_arr[n];
    double d[n];
    double e[n];
    double ldetc;

    fill(L, 1.0, n);
    /* Get the determinant of C^-1 and solve C a = L */
    ldetc = snsolve_retdet(t, err2, var, tcorr, a, L, n);
    /* Calculate Cq = 1/(L^T C^-1 L) */
    double Cq = 1/dot(L, a, n);
    /* solve C b = y */
    snsolve(t, err2, var, tcorr, b, y, n);
    double c = dot(L, b, n);
    c *= Cq;

    fill(c_arr, c, n);
    snsolve(t, err2, var, tcorr, d, c_arr, n);
    ewise_subtract(e, b, d, n);
    double chisq = dot(y, e, n);
    
    return chisq;
}

void fill(double *L, double val, int n) {
    for (int i = 0; i < n; i++) {
        L[i] = val;
    }
}

double dot(double *a, double *b, int n) {
    double sum = 0;
    
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

void ewise_subtract(double *out, double *a, double *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] - b[i];
    }
}