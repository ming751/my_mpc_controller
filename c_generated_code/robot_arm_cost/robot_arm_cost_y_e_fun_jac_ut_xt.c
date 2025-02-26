/* This file was automatically generated by CasADi 3.6.7.
 *  It consists of: 
 *   1) content generated by CasADi runtime: not copyrighted
 *   2) template code copied from CasADi source: permissively licensed (MIT-0)
 *   3) user code: owned by the user
 *
 */
#ifdef __cplusplus
extern "C" {
#endif

/* How to prefix internal symbols */
#ifdef CASADI_CODEGEN_PREFIX
  #define CASADI_NAMESPACE_CONCAT(NS, ID) _CASADI_NAMESPACE_CONCAT(NS, ID)
  #define _CASADI_NAMESPACE_CONCAT(NS, ID) NS ## ID
  #define CASADI_PREFIX(ID) CASADI_NAMESPACE_CONCAT(CODEGEN_PREFIX, ID)
#else
  #define CASADI_PREFIX(ID) robot_arm_cost_y_e_fun_jac_ut_xt_ ## ID
#endif

#include <math.h>

#ifndef casadi_real
#define casadi_real double
#endif

#ifndef casadi_int
#define casadi_int int
#endif

/* Add prefix to internal symbols */
#define casadi_f0 CASADI_PREFIX(f0)
#define casadi_s0 CASADI_PREFIX(s0)
#define casadi_s1 CASADI_PREFIX(s1)
#define casadi_s2 CASADI_PREFIX(s2)
#define casadi_s3 CASADI_PREFIX(s3)
#define casadi_s4 CASADI_PREFIX(s4)
#define casadi_s5 CASADI_PREFIX(s5)

/* Symbol visibility in DLLs */
#ifndef CASADI_SYMBOL_EXPORT
  #if defined(_WIN32) || defined(__WIN32__) || defined(__CYGWIN__)
    #if defined(STATIC_LINKED)
      #define CASADI_SYMBOL_EXPORT
    #else
      #define CASADI_SYMBOL_EXPORT __declspec(dllexport)
    #endif
  #elif defined(__GNUC__) && defined(GCC_HASCLASSVISIBILITY)
    #define CASADI_SYMBOL_EXPORT __attribute__ ((visibility ("default")))
  #else
    #define CASADI_SYMBOL_EXPORT
  #endif
#endif

static const casadi_int casadi_s0[12] = {8, 1, 0, 8, 0, 1, 2, 3, 4, 5, 6, 7};
static const casadi_int casadi_s1[3] = {0, 0, 0};
static const casadi_int casadi_s2[4] = {0, 1, 0, 0};
static const casadi_int casadi_s3[7] = {3, 1, 0, 3, 0, 1, 2};
static const casadi_int casadi_s4[18] = {8, 3, 0, 4, 8, 12, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
static const casadi_int casadi_s5[3] = {3, 0, 0};

/* robot_arm_cost_y_e_fun_jac_ut_xt:(i0[8],i1[],i2[0],i3[],i4[3])->(o0[3],o1[8x3,12nz],o2[3x0]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a8, a9;
  a0=2.9836803753664046e-02;
  a1=-1.8402900000000001e-03;
  a2=9.9804914959096003e-01;
  a3=arg[0]? arg[0][0] : 0;
  a4=cos(a3);
  a5=(a2*a4);
  a6=-6.2433124191511064e-02;
  a3=sin(a3);
  a7=(a6*a3);
  a5=(a5-a7);
  a7=(a1*a5);
  a8=1.0999999999999999e-02;
  a2=(a2*a3);
  a6=(a6*a4);
  a2=(a2+a6);
  a6=(a8*a2);
  a9=8.0152940179646143e-08;
  a6=(a6+a9);
  a7=(a7+a6);
  a0=(a0+a7);
  a7=8.9999999999999998e-04;
  a6=(a7*a5);
  a9=1.0910800000000000e-01;
  a10=9.8251889856312924e-01;
  a11=arg[0]? arg[0][1] : 0;
  a12=cos(a11);
  a13=(a10*a12);
  a14=1.8616286946191887e-01;
  a11=sin(a11);
  a15=(a14*a11);
  a16=(a13+a15);
  a17=(a2*a16);
  a18=2.1092878994643724e-06;
  a19=-1.8616286946191887e-01;
  a20=(a19*a12);
  a10=(a10*a11);
  a21=(a20+a10);
  a22=(a18*a21);
  a17=(a17+a22);
  a22=(a9*a17);
  a23=-4.2289899999999998e-02;
  a14=(a14*a12);
  a14=(a14-a10);
  a12=(a2*a14);
  a19=(a19*a11);
  a19=(a13-a19);
  a11=(a18*a19);
  a12=(a12+a11);
  a11=(a23*a12);
  a22=(a22+a11);
  a6=(a6+a22);
  a0=(a0+a6);
  a6=-1.5528400000000000e-03;
  a22=1.5831900375320496e-01;
  a11=arg[0]? arg[0][2] : 0;
  a24=cos(a11);
  a25=(a22*a24);
  a26=-9.8738801544811317e-01;
  a11=sin(a11);
  a27=(a26*a11);
  a28=(a25+a27);
  a29=(a5*a28);
  a30=4.4937245474095094e-01;
  a31=(a30*a24);
  a32=7.2052776132521146e-02;
  a33=(a32*a11);
  a34=(a31+a33);
  a35=(a17*a34);
  a36=8.7920389556159528e-01;
  a37=(a36*a24);
  a38=1.4097270800063794e-01;
  a39=(a38*a11);
  a40=(a37+a39);
  a41=(a12*a40);
  a35=(a35+a41);
  a29=(a29+a35);
  a35=(a6*a29);
  a41=3.8560800000000001e-04;
  a26=(a26*a24);
  a22=(a22*a11);
  a26=(a26-a22);
  a22=(a5*a26);
  a32=(a32*a24);
  a30=(a30*a11);
  a32=(a32-a30);
  a30=(a17*a32);
  a38=(a38*a24);
  a36=(a36*a11);
  a38=(a38-a36);
  a36=(a12*a38);
  a30=(a30+a36);
  a22=(a22+a30);
  a30=(a41*a22);
  a36=-1.2670000000000001e-01;
  a11=1.7038398586466386e-07;
  a24=(a11*a5);
  a42=-8.9043404829986628e-01;
  a43=(a42*a17);
  a44=4.5511230001866820e-01;
  a45=(a44*a12);
  a43=(a43+a45);
  a24=(a24+a43);
  a43=(a36*a24);
  a30=(a30+a43);
  a35=(a35+a30);
  a0=(a0+a35);
  a35=-2.7000000000000002e-01;
  a30=2.4087403938522856e-01;
  a43=arg[0]? arg[0][3] : 0;
  a45=sin(a43);
  a46=(a30*a45);
  a47=-7.9534806859374307e-03;
  a43=cos(a43);
  a48=(a47*a43);
  a46=(a46+a48);
  a48=(a29*a46);
  a49=9.6999516239483097e-01;
  a50=(a49*a45);
  a51=-3.2028490845610859e-02;
  a52=(a51*a43);
  a50=(a50+a52);
  a52=(a22*a50);
  a53=3.3001243626095737e-02;
  a54=(a53*a45);
  a55=9.9945531061630322e-01;
  a56=(a55*a43);
  a54=(a54+a56);
  a56=(a24*a54);
  a52=(a52+a56);
  a48=(a48+a52);
  a48=(a35*a48);
  a0=(a0+a48);
  a48=arg[4]? arg[4][0] : 0;
  a0=(a0-a48);
  if (res[0]!=0) res[0][0]=a0;
  a0=1.3002618808142252e-01;
  a48=4.4635152763800194e-02;
  a52=(a48*a4);
  a56=7.1355633079580183e-01;
  a57=(a56*a3);
  a52=(a52-a57);
  a57=(a1*a52);
  a48=(a48*a3);
  a56=(a56*a4);
  a48=(a48+a56);
  a56=(a8*a48);
  a58=2.6568633295429644e-02;
  a56=(a56+a58);
  a57=(a57+a56);
  a0=(a0+a57);
  a57=(a7*a52);
  a56=(a48*a16);
  a58=6.9917456040604331e-01;
  a59=(a58*a21);
  a56=(a56+a59);
  a59=(a9*a56);
  a60=(a48*a14);
  a61=(a58*a19);
  a60=(a60+a61);
  a61=(a23*a60);
  a59=(a59+a61);
  a57=(a57+a59);
  a0=(a0+a57);
  a57=(a52*a28);
  a59=(a56*a34);
  a61=(a60*a40);
  a59=(a59+a61);
  a57=(a57+a59);
  a59=(a6*a57);
  a61=(a52*a26);
  a62=(a56*a32);
  a63=(a60*a38);
  a62=(a62+a63);
  a61=(a61+a62);
  a62=(a41*a61);
  a63=(a11*a52);
  a64=(a42*a56);
  a65=(a44*a60);
  a64=(a64+a65);
  a63=(a63+a64);
  a64=(a36*a63);
  a62=(a62+a64);
  a59=(a59+a62);
  a0=(a0+a59);
  a59=(a57*a46);
  a62=(a61*a50);
  a64=(a63*a54);
  a62=(a62+a64);
  a59=(a59+a62);
  a59=(a35*a59);
  a0=(a0+a59);
  a59=arg[4]? arg[4][1] : 0;
  a0=(a0-a59);
  if (res[0]!=0) res[0][1]=a0;
  a0=1.3266405326240711e+00;
  a59=-4.3653157257109741e-02;
  a62=(a59*a4);
  a64=-6.9781048128049727e-01;
  a65=(a64*a3);
  a62=(a62-a65);
  a65=(a1*a62);
  a59=(a59*a3);
  a64=(a64*a4);
  a59=(a59+a64);
  a64=(a8*a59);
  a4=2.7168138044528597e-02;
  a64=(a64+a4);
  a65=(a65+a64);
  a0=(a0+a65);
  a65=(a7*a62);
  a64=(a59*a16);
  a4=7.1495100117180521e-01;
  a21=(a4*a21);
  a64=(a64+a21);
  a21=(a9*a64);
  a3=(a59*a14);
  a19=(a4*a19);
  a3=(a3+a19);
  a19=(a23*a3);
  a21=(a21+a19);
  a65=(a65+a21);
  a0=(a0+a65);
  a65=(a62*a28);
  a21=(a64*a34);
  a19=(a3*a40);
  a21=(a21+a19);
  a65=(a65+a21);
  a21=(a6*a65);
  a19=(a62*a26);
  a66=(a64*a32);
  a67=(a3*a38);
  a66=(a66+a67);
  a19=(a19+a66);
  a66=(a41*a19);
  a67=(a11*a62);
  a68=(a42*a64);
  a69=(a44*a3);
  a68=(a68+a69);
  a67=(a67+a68);
  a68=(a36*a67);
  a66=(a66+a68);
  a21=(a21+a66);
  a0=(a0+a21);
  a21=(a65*a46);
  a66=(a19*a50);
  a68=(a67*a54);
  a66=(a66+a68);
  a21=(a21+a66);
  a21=(a35*a21);
  a0=(a0+a21);
  a21=arg[4]? arg[4][2] : 0;
  a0=(a0-a21);
  if (res[0]!=0) res[0][2]=a0;
  a0=(a8*a5);
  a21=(a1*a2);
  a0=(a0-a21);
  a21=(a16*a5);
  a66=(a9*a21);
  a68=(a14*a5);
  a69=(a23*a68);
  a66=(a66+a69);
  a69=(a7*a2);
  a66=(a66-a69);
  a0=(a0+a66);
  a66=(a34*a21);
  a69=(a40*a68);
  a66=(a66+a69);
  a69=(a28*a2);
  a66=(a66-a69);
  a69=(a6*a66);
  a70=(a32*a21);
  a71=(a38*a68);
  a70=(a70+a71);
  a71=(a26*a2);
  a70=(a70-a71);
  a71=(a41*a70);
  a21=(a42*a21);
  a68=(a44*a68);
  a21=(a21+a68);
  a68=(a11*a2);
  a21=(a21-a68);
  a68=(a36*a21);
  a71=(a71+a68);
  a69=(a69+a71);
  a0=(a0+a69);
  a66=(a46*a66);
  a70=(a50*a70);
  a21=(a54*a21);
  a70=(a70+a21);
  a66=(a66+a70);
  a66=(a35*a66);
  a0=(a0+a66);
  if (res[1]!=0) res[1][0]=a0;
  a0=(a9*a12);
  a15=(a15+a13);
  a2=(a2*a15);
  a10=(a10+a20);
  a18=(a18*a10);
  a2=(a2+a18);
  a18=(a23*a2);
  a0=(a0-a18);
  a18=(a34*a12);
  a20=(a40*a2);
  a18=(a18-a20);
  a20=(a6*a18);
  a13=(a32*a12);
  a66=(a38*a2);
  a13=(a13-a66);
  a66=(a41*a13);
  a70=(a42*a12);
  a2=(a44*a2);
  a70=(a70-a2);
  a2=(a36*a70);
  a66=(a66+a2);
  a20=(a20+a66);
  a0=(a0+a20);
  a18=(a46*a18);
  a13=(a50*a13);
  a70=(a54*a70);
  a13=(a13+a70);
  a18=(a18+a13);
  a18=(a35*a18);
  a0=(a0+a18);
  if (res[1]!=0) res[1][1]=a0;
  a0=(a6*a22);
  a27=(a27+a25);
  a5=(a5*a27);
  a33=(a33+a31);
  a17=(a17*a33);
  a39=(a39+a37);
  a12=(a12*a39);
  a17=(a17+a12);
  a5=(a5+a17);
  a17=(a41*a5);
  a0=(a0-a17);
  a17=(a46*a22);
  a5=(a50*a5);
  a17=(a17-a5);
  a17=(a35*a17);
  a0=(a0+a17);
  if (res[1]!=0) res[1][2]=a0;
  a30=(a30*a43);
  a47=(a47*a45);
  a30=(a30-a47);
  a29=(a29*a30);
  a49=(a49*a43);
  a51=(a51*a45);
  a49=(a49-a51);
  a22=(a22*a49);
  a53=(a53*a43);
  a55=(a55*a45);
  a53=(a53-a55);
  a24=(a24*a53);
  a22=(a22+a24);
  a29=(a29+a22);
  a29=(a35*a29);
  if (res[1]!=0) res[1][3]=a29;
  a29=(a8*a52);
  a22=(a1*a48);
  a29=(a29-a22);
  a22=(a16*a52);
  a24=(a9*a22);
  a55=(a14*a52);
  a45=(a23*a55);
  a24=(a24+a45);
  a45=(a7*a48);
  a24=(a24-a45);
  a29=(a29+a24);
  a24=(a34*a22);
  a45=(a40*a55);
  a24=(a24+a45);
  a45=(a28*a48);
  a24=(a24-a45);
  a45=(a6*a24);
  a43=(a32*a22);
  a51=(a38*a55);
  a43=(a43+a51);
  a51=(a26*a48);
  a43=(a43-a51);
  a51=(a41*a43);
  a22=(a42*a22);
  a55=(a44*a55);
  a22=(a22+a55);
  a55=(a11*a48);
  a22=(a22-a55);
  a55=(a36*a22);
  a51=(a51+a55);
  a45=(a45+a51);
  a29=(a29+a45);
  a24=(a46*a24);
  a43=(a50*a43);
  a22=(a54*a22);
  a43=(a43+a22);
  a24=(a24+a43);
  a24=(a35*a24);
  a29=(a29+a24);
  if (res[1]!=0) res[1][4]=a29;
  a29=(a9*a60);
  a48=(a48*a15);
  a58=(a58*a10);
  a48=(a48+a58);
  a58=(a23*a48);
  a29=(a29-a58);
  a58=(a34*a60);
  a24=(a40*a48);
  a58=(a58-a24);
  a24=(a6*a58);
  a43=(a32*a60);
  a22=(a38*a48);
  a43=(a43-a22);
  a22=(a41*a43);
  a45=(a42*a60);
  a48=(a44*a48);
  a45=(a45-a48);
  a48=(a36*a45);
  a22=(a22+a48);
  a24=(a24+a22);
  a29=(a29+a24);
  a58=(a46*a58);
  a43=(a50*a43);
  a45=(a54*a45);
  a43=(a43+a45);
  a58=(a58+a43);
  a58=(a35*a58);
  a29=(a29+a58);
  if (res[1]!=0) res[1][5]=a29;
  a29=(a6*a61);
  a52=(a52*a27);
  a56=(a56*a33);
  a60=(a60*a39);
  a56=(a56+a60);
  a52=(a52+a56);
  a56=(a41*a52);
  a29=(a29-a56);
  a56=(a46*a61);
  a52=(a50*a52);
  a56=(a56-a52);
  a56=(a35*a56);
  a29=(a29+a56);
  if (res[1]!=0) res[1][6]=a29;
  a57=(a57*a30);
  a61=(a61*a49);
  a63=(a63*a53);
  a61=(a61+a63);
  a57=(a57+a61);
  a57=(a35*a57);
  if (res[1]!=0) res[1][7]=a57;
  a8=(a8*a62);
  a1=(a1*a59);
  a8=(a8-a1);
  a16=(a16*a62);
  a1=(a9*a16);
  a14=(a14*a62);
  a57=(a23*a14);
  a1=(a1+a57);
  a7=(a7*a59);
  a1=(a1-a7);
  a8=(a8+a1);
  a1=(a34*a16);
  a7=(a40*a14);
  a1=(a1+a7);
  a28=(a28*a59);
  a1=(a1-a28);
  a28=(a6*a1);
  a7=(a32*a16);
  a57=(a38*a14);
  a7=(a7+a57);
  a26=(a26*a59);
  a7=(a7-a26);
  a26=(a41*a7);
  a16=(a42*a16);
  a14=(a44*a14);
  a16=(a16+a14);
  a11=(a11*a59);
  a16=(a16-a11);
  a11=(a36*a16);
  a26=(a26+a11);
  a28=(a28+a26);
  a8=(a8+a28);
  a1=(a46*a1);
  a7=(a50*a7);
  a16=(a54*a16);
  a7=(a7+a16);
  a1=(a1+a7);
  a1=(a35*a1);
  a8=(a8+a1);
  if (res[1]!=0) res[1][8]=a8;
  a9=(a9*a3);
  a59=(a59*a15);
  a4=(a4*a10);
  a59=(a59+a4);
  a23=(a23*a59);
  a9=(a9-a23);
  a34=(a34*a3);
  a40=(a40*a59);
  a34=(a34-a40);
  a40=(a6*a34);
  a32=(a32*a3);
  a38=(a38*a59);
  a32=(a32-a38);
  a38=(a41*a32);
  a42=(a42*a3);
  a44=(a44*a59);
  a42=(a42-a44);
  a36=(a36*a42);
  a38=(a38+a36);
  a40=(a40+a38);
  a9=(a9+a40);
  a34=(a46*a34);
  a32=(a50*a32);
  a54=(a54*a42);
  a32=(a32+a54);
  a34=(a34+a32);
  a34=(a35*a34);
  a9=(a9+a34);
  if (res[1]!=0) res[1][9]=a9;
  a6=(a6*a19);
  a62=(a62*a27);
  a64=(a64*a33);
  a3=(a3*a39);
  a64=(a64+a3);
  a62=(a62+a64);
  a41=(a41*a62);
  a6=(a6-a41);
  a46=(a46*a19);
  a50=(a50*a62);
  a46=(a46-a50);
  a46=(a35*a46);
  a6=(a6+a46);
  if (res[1]!=0) res[1][10]=a6;
  a65=(a65*a30);
  a19=(a19*a49);
  a67=(a67*a53);
  a19=(a19+a67);
  a65=(a65+a19);
  a35=(a35*a65);
  if (res[1]!=0) res[1][11]=a35;
  return 0;
}

CASADI_SYMBOL_EXPORT int robot_arm_cost_y_e_fun_jac_ut_xt(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int robot_arm_cost_y_e_fun_jac_ut_xt_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int robot_arm_cost_y_e_fun_jac_ut_xt_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void robot_arm_cost_y_e_fun_jac_ut_xt_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int robot_arm_cost_y_e_fun_jac_ut_xt_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void robot_arm_cost_y_e_fun_jac_ut_xt_release(int mem) {
}

CASADI_SYMBOL_EXPORT void robot_arm_cost_y_e_fun_jac_ut_xt_incref(void) {
}

CASADI_SYMBOL_EXPORT void robot_arm_cost_y_e_fun_jac_ut_xt_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int robot_arm_cost_y_e_fun_jac_ut_xt_n_in(void) { return 5;}

CASADI_SYMBOL_EXPORT casadi_int robot_arm_cost_y_e_fun_jac_ut_xt_n_out(void) { return 3;}

CASADI_SYMBOL_EXPORT casadi_real robot_arm_cost_y_e_fun_jac_ut_xt_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* robot_arm_cost_y_e_fun_jac_ut_xt_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* robot_arm_cost_y_e_fun_jac_ut_xt_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    case 1: return "o1";
    case 2: return "o2";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* robot_arm_cost_y_e_fun_jac_ut_xt_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s1;
    case 4: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* robot_arm_cost_y_e_fun_jac_ut_xt_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s3;
    case 1: return casadi_s4;
    case 2: return casadi_s5;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int robot_arm_cost_y_e_fun_jac_ut_xt_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5;
  if (sz_res) *sz_res = 3;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int robot_arm_cost_y_e_fun_jac_ut_xt_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 5*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 3*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
