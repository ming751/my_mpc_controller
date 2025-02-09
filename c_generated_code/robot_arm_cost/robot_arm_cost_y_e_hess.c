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
  #define CASADI_PREFIX(ID) robot_arm_cost_y_e_hess_ ## ID
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
static const casadi_int casadi_s4[27] = {8, 8, 0, 4, 8, 12, 16, 16, 16, 16, 16, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};

/* robot_arm_cost_y_e_hess:(i0[8],i1[],i2[0],i3[3],i4[],i5[3])->(o0[8x8,16nz]) */
static int casadi_f0(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem) {
  casadi_real a0, a1, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a2, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a3, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a4, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a5, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a6, a60, a61, a62, a63, a64, a65, a66, a67, a68, a69, a7, a70, a71, a72, a73, a74, a75, a76, a77, a78, a79, a8, a80, a81, a82, a83, a84, a85, a86, a9;
  a0=-4.3653157257109741e-02;
  a1=1.8616286946191887e-01;
  a2=arg[0]? arg[0][1] : 0;
  a3=cos(a2);
  a4=(a1*a3);
  a5=9.8251889856312924e-01;
  a2=sin(a2);
  a6=(a5*a2);
  a4=(a4-a6);
  a7=4.5511230001866820e-01;
  a8=3.3001243626095737e-02;
  a9=arg[0]? arg[0][3] : 0;
  a10=sin(a9);
  a11=(a8*a10);
  a12=9.9945531061630322e-01;
  a9=cos(a9);
  a13=(a12*a9);
  a11=(a11+a13);
  a13=-2.7000000000000002e-01;
  a14=arg[3]? arg[3][2] : 0;
  a15=(a13*a14);
  a16=(a11*a15);
  a17=-1.2670000000000001e-01;
  a18=(a17*a14);
  a16=(a16+a18);
  a18=(a7*a16);
  a19=1.4097270800063794e-01;
  a20=arg[0]? arg[0][2] : 0;
  a21=cos(a20);
  a22=(a19*a21);
  a23=8.7920389556159528e-01;
  a20=sin(a20);
  a24=(a23*a20);
  a22=(a22-a24);
  a24=9.6999516239483097e-01;
  a25=(a24*a10);
  a26=-3.2028490845610859e-02;
  a27=(a26*a9);
  a25=(a25+a27);
  a27=(a25*a15);
  a28=3.8560800000000001e-04;
  a29=(a28*a14);
  a27=(a27+a29);
  a29=(a22*a27);
  a18=(a18+a29);
  a29=(a23*a21);
  a30=(a19*a20);
  a31=(a29+a30);
  a32=2.4087403938522856e-01;
  a33=(a32*a10);
  a34=-7.9534806859374307e-03;
  a35=(a34*a9);
  a33=(a33+a35);
  a35=(a33*a15);
  a36=-1.5528400000000000e-03;
  a37=(a36*a14);
  a35=(a35+a37);
  a37=(a31*a35);
  a18=(a18+a37);
  a37=-4.2289899999999998e-02;
  a38=(a37*a14);
  a18=(a18+a38);
  a38=(a4*a18);
  a39=(a5*a3);
  a40=(a1*a2);
  a41=(a39+a40);
  a42=-8.9043404829986628e-01;
  a43=(a42*a16);
  a44=7.2052776132521146e-02;
  a45=(a44*a21);
  a46=4.4937245474095094e-01;
  a47=(a46*a20);
  a45=(a45-a47);
  a47=(a45*a27);
  a43=(a43+a47);
  a47=(a46*a21);
  a48=(a44*a20);
  a49=(a47+a48);
  a50=(a49*a35);
  a43=(a43+a50);
  a50=1.0910800000000000e-01;
  a51=(a50*a14);
  a43=(a43+a51);
  a51=(a41*a43);
  a38=(a38+a51);
  a51=1.0999999999999999e-02;
  a52=(a51*a14);
  a38=(a38+a52);
  a52=(a0*a38);
  a53=-6.9781048128049727e-01;
  a54=1.7038398586466386e-07;
  a16=(a54*a16);
  a55=-9.8738801544811317e-01;
  a56=(a55*a21);
  a57=1.5831900375320496e-01;
  a58=(a57*a20);
  a56=(a56-a58);
  a58=(a56*a27);
  a16=(a16+a58);
  a58=(a57*a21);
  a59=(a55*a20);
  a60=(a58+a59);
  a61=(a60*a35);
  a16=(a16+a61);
  a61=8.9999999999999998e-04;
  a62=(a61*a14);
  a16=(a16+a62);
  a62=-1.8402900000000001e-03;
  a14=(a62*a14);
  a16=(a16+a14);
  a14=(a53*a16);
  a52=(a52-a14);
  a14=4.4635152763800194e-02;
  a63=arg[3]? arg[3][1] : 0;
  a64=(a13*a63);
  a65=(a11*a64);
  a66=(a17*a63);
  a65=(a65+a66);
  a66=(a7*a65);
  a67=(a25*a64);
  a68=(a28*a63);
  a67=(a67+a68);
  a68=(a22*a67);
  a66=(a66+a68);
  a68=(a33*a64);
  a69=(a36*a63);
  a68=(a68+a69);
  a69=(a31*a68);
  a66=(a66+a69);
  a69=(a37*a63);
  a66=(a66+a69);
  a69=(a4*a66);
  a70=(a42*a65);
  a71=(a45*a67);
  a70=(a70+a71);
  a71=(a49*a68);
  a70=(a70+a71);
  a71=(a50*a63);
  a70=(a70+a71);
  a71=(a41*a70);
  a69=(a69+a71);
  a71=(a51*a63);
  a69=(a69+a71);
  a71=(a14*a69);
  a52=(a52+a71);
  a71=7.1355633079580183e-01;
  a65=(a54*a65);
  a72=(a56*a67);
  a65=(a65+a72);
  a72=(a60*a68);
  a65=(a65+a72);
  a72=(a61*a63);
  a65=(a65+a72);
  a63=(a62*a63);
  a65=(a65+a63);
  a63=(a71*a65);
  a52=(a52-a63);
  a63=9.9804914959096003e-01;
  a72=arg[3]? arg[3][0] : 0;
  a13=(a13*a72);
  a11=(a11*a13);
  a17=(a17*a72);
  a11=(a11+a17);
  a17=(a7*a11);
  a25=(a25*a13);
  a28=(a28*a72);
  a25=(a25+a28);
  a28=(a22*a25);
  a17=(a17+a28);
  a33=(a33*a13);
  a36=(a36*a72);
  a33=(a33+a36);
  a36=(a31*a33);
  a17=(a17+a36);
  a37=(a37*a72);
  a17=(a17+a37);
  a37=(a4*a17);
  a36=(a42*a11);
  a28=(a45*a25);
  a36=(a36+a28);
  a28=(a49*a33);
  a36=(a36+a28);
  a50=(a50*a72);
  a36=(a36+a50);
  a50=(a41*a36);
  a37=(a37+a50);
  a51=(a51*a72);
  a37=(a37+a51);
  a51=(a63*a37);
  a52=(a52+a51);
  a51=-6.2433124191511064e-02;
  a11=(a54*a11);
  a50=(a56*a25);
  a11=(a11+a50);
  a50=(a60*a33);
  a11=(a11+a50);
  a61=(a61*a72);
  a11=(a11+a61);
  a62=(a62*a72);
  a11=(a11+a62);
  a62=(a51*a11);
  a52=(a52-a62);
  a62=arg[0]? arg[0][0] : 0;
  a72=sin(a62);
  a52=(a52*a72);
  a38=(a53*a38);
  a16=(a0*a16);
  a38=(a38+a16);
  a69=(a71*a69);
  a38=(a38+a69);
  a65=(a14*a65);
  a38=(a38+a65);
  a37=(a51*a37);
  a38=(a38+a37);
  a11=(a63*a11);
  a38=(a38+a11);
  a62=cos(a62);
  a38=(a38*a62);
  a52=(a52+a38);
  a52=(-a52);
  if (res[0]!=0) res[0][0]=a52;
  a52=(a43*a4);
  a40=(a40+a39);
  a38=(a18*a40);
  a52=(a52-a38);
  a38=(a0*a52);
  a11=(a70*a4);
  a37=(a66*a40);
  a11=(a11-a37);
  a37=(a14*a11);
  a38=(a38+a37);
  a37=(a36*a4);
  a40=(a17*a40);
  a37=(a37-a40);
  a40=(a63*a37);
  a38=(a38+a40);
  a38=(a62*a38);
  a52=(a53*a52);
  a11=(a71*a11);
  a52=(a52+a11);
  a37=(a51*a37);
  a52=(a52+a37);
  a52=(a72*a52);
  a38=(a38-a52);
  if (res[0]!=0) res[0][1]=a38;
  a52=(a35*a22);
  a30=(a30+a29);
  a29=(a27*a30);
  a52=(a52-a29);
  a29=(a4*a52);
  a37=(a35*a45);
  a48=(a48+a47);
  a47=(a27*a48);
  a37=(a37-a47);
  a47=(a41*a37);
  a29=(a29+a47);
  a47=(a0*a29);
  a11=(a35*a56);
  a59=(a59+a58);
  a58=(a27*a59);
  a11=(a11-a58);
  a58=(a53*a11);
  a47=(a47-a58);
  a58=(a68*a22);
  a40=(a67*a30);
  a58=(a58-a40);
  a40=(a4*a58);
  a65=(a68*a45);
  a69=(a67*a48);
  a65=(a65-a69);
  a69=(a41*a65);
  a40=(a40+a69);
  a69=(a14*a40);
  a47=(a47+a69);
  a69=(a68*a56);
  a16=(a67*a59);
  a69=(a69-a16);
  a16=(a71*a69);
  a47=(a47-a16);
  a16=(a33*a22);
  a30=(a25*a30);
  a16=(a16-a30);
  a30=(a4*a16);
  a61=(a33*a45);
  a48=(a25*a48);
  a61=(a61-a48);
  a48=(a41*a61);
  a30=(a30+a48);
  a48=(a63*a30);
  a47=(a47+a48);
  a48=(a33*a56);
  a59=(a25*a59);
  a48=(a48-a59);
  a59=(a51*a48);
  a47=(a47-a59);
  a47=(a62*a47);
  a29=(a53*a29);
  a11=(a0*a11);
  a29=(a29+a11);
  a40=(a71*a40);
  a29=(a29+a40);
  a69=(a14*a69);
  a29=(a29+a69);
  a30=(a51*a30);
  a29=(a29+a30);
  a48=(a63*a48);
  a29=(a29+a48);
  a29=(a72*a29);
  a47=(a47-a29);
  if (res[0]!=0) res[0][2]=a47;
  a29=(a8*a9);
  a48=(a12*a10);
  a29=(a29-a48);
  a48=(a15*a29);
  a30=(a7*a48);
  a69=(a24*a9);
  a40=(a26*a10);
  a69=(a69-a40);
  a40=(a15*a69);
  a11=(a22*a40);
  a30=(a30+a11);
  a11=(a32*a9);
  a59=(a34*a10);
  a11=(a11-a59);
  a59=(a15*a11);
  a50=(a31*a59);
  a30=(a30+a50);
  a50=(a4*a30);
  a28=(a42*a48);
  a73=(a45*a40);
  a28=(a28+a73);
  a73=(a49*a59);
  a28=(a28+a73);
  a73=(a41*a28);
  a50=(a50+a73);
  a73=(a0*a50);
  a48=(a54*a48);
  a74=(a56*a40);
  a48=(a48+a74);
  a74=(a60*a59);
  a48=(a48+a74);
  a74=(a53*a48);
  a73=(a73-a74);
  a74=(a64*a29);
  a75=(a7*a74);
  a76=(a64*a69);
  a77=(a22*a76);
  a75=(a75+a77);
  a77=(a64*a11);
  a78=(a31*a77);
  a75=(a75+a78);
  a78=(a4*a75);
  a79=(a42*a74);
  a80=(a45*a76);
  a79=(a79+a80);
  a80=(a49*a77);
  a79=(a79+a80);
  a80=(a41*a79);
  a78=(a78+a80);
  a80=(a14*a78);
  a73=(a73+a80);
  a74=(a54*a74);
  a80=(a56*a76);
  a74=(a74+a80);
  a80=(a60*a77);
  a74=(a74+a80);
  a80=(a71*a74);
  a73=(a73-a80);
  a29=(a13*a29);
  a80=(a7*a29);
  a69=(a13*a69);
  a81=(a22*a69);
  a80=(a80+a81);
  a11=(a13*a11);
  a81=(a31*a11);
  a80=(a80+a81);
  a81=(a4*a80);
  a82=(a42*a29);
  a83=(a45*a69);
  a82=(a82+a83);
  a83=(a49*a11);
  a82=(a82+a83);
  a83=(a41*a82);
  a81=(a81+a83);
  a83=(a63*a81);
  a73=(a73+a83);
  a29=(a54*a29);
  a83=(a56*a69);
  a29=(a29+a83);
  a83=(a60*a11);
  a29=(a29+a83);
  a83=(a51*a29);
  a73=(a73-a83);
  a73=(a62*a73);
  a50=(a53*a50);
  a48=(a0*a48);
  a50=(a50+a48);
  a78=(a71*a78);
  a50=(a50+a78);
  a74=(a14*a74);
  a50=(a50+a74);
  a81=(a51*a81);
  a50=(a50+a81);
  a29=(a63*a29);
  a50=(a50+a29);
  a50=(a72*a50);
  a73=(a73-a50);
  if (res[0]!=0) res[0][3]=a73;
  if (res[0]!=0) res[0][4]=a38;
  a38=7.1495100117180521e-01;
  a50=(a38*a43);
  a29=6.9917456040604331e-01;
  a81=(a29*a70);
  a50=(a50+a81);
  a81=2.1092878994643724e-06;
  a74=(a81*a36);
  a50=(a50+a74);
  a74=(a5*a50);
  a78=-1.8616286946191887e-01;
  a48=(a38*a18);
  a83=(a29*a66);
  a48=(a48+a83);
  a83=(a81*a17);
  a48=(a48+a83);
  a83=(a78*a48);
  a84=(a0*a72);
  a85=(a53*a62);
  a84=(a84+a85);
  a18=(a84*a18);
  a85=(a14*a72);
  a86=(a71*a62);
  a85=(a85+a86);
  a66=(a85*a66);
  a18=(a18+a66);
  a66=(a63*a72);
  a86=(a51*a62);
  a66=(a66+a86);
  a17=(a66*a17);
  a18=(a18+a17);
  a17=(a5*a18);
  a83=(a83+a17);
  a74=(a74-a83);
  a43=(a84*a43);
  a70=(a85*a70);
  a43=(a43+a70);
  a36=(a66*a36);
  a43=(a43+a36);
  a36=(a1*a43);
  a74=(a74+a36);
  a74=(a74*a2);
  a48=(a5*a48);
  a18=(a1*a18);
  a48=(a48+a18);
  a50=(a78*a50);
  a48=(a48+a50);
  a43=(a5*a43);
  a48=(a48+a43);
  a48=(a48*a3);
  a74=(a74+a48);
  a74=(-a74);
  if (res[0]!=0) res[0][5]=a74;
  a74=(a38*a37);
  a48=(a29*a65);
  a74=(a74+a48);
  a48=(a81*a61);
  a74=(a74+a48);
  a48=(a5*a74);
  a43=(a38*a52);
  a50=(a29*a58);
  a43=(a43+a50);
  a50=(a81*a16);
  a43=(a43+a50);
  a50=(a78*a43);
  a52=(a84*a52);
  a58=(a85*a58);
  a52=(a52+a58);
  a16=(a66*a16);
  a52=(a52+a16);
  a16=(a5*a52);
  a50=(a50+a16);
  a48=(a48-a50);
  a37=(a84*a37);
  a65=(a85*a65);
  a37=(a37+a65);
  a61=(a66*a61);
  a37=(a37+a61);
  a61=(a1*a37);
  a48=(a48+a61);
  a48=(a3*a48);
  a43=(a5*a43);
  a52=(a1*a52);
  a43=(a43+a52);
  a74=(a78*a74);
  a43=(a43+a74);
  a37=(a5*a37);
  a43=(a43+a37);
  a43=(a2*a43);
  a48=(a48-a43);
  if (res[0]!=0) res[0][6]=a48;
  a43=(a38*a28);
  a37=(a29*a79);
  a43=(a43+a37);
  a37=(a81*a82);
  a43=(a43+a37);
  a37=(a5*a43);
  a74=(a38*a30);
  a52=(a29*a75);
  a74=(a74+a52);
  a52=(a81*a80);
  a74=(a74+a52);
  a52=(a78*a74);
  a30=(a84*a30);
  a75=(a85*a75);
  a30=(a30+a75);
  a80=(a66*a80);
  a30=(a30+a80);
  a80=(a5*a30);
  a52=(a52+a80);
  a37=(a37-a52);
  a28=(a84*a28);
  a79=(a85*a79);
  a28=(a28+a79);
  a82=(a66*a82);
  a28=(a28+a82);
  a82=(a1*a28);
  a37=(a37+a82);
  a37=(a3*a37);
  a74=(a5*a74);
  a1=(a1*a30);
  a74=(a74+a1);
  a43=(a78*a43);
  a74=(a74+a43);
  a5=(a5*a28);
  a74=(a74+a5);
  a74=(a2*a74);
  a37=(a37-a74);
  if (res[0]!=0) res[0][7]=a37;
  if (res[0]!=0) res[0][8]=a47;
  if (res[0]!=0) res[0][9]=a48;
  a48=(a84*a4);
  a2=(a78*a2);
  a39=(a39-a2);
  a2=(a38*a39);
  a48=(a48+a2);
  a2=(a48*a35);
  a47=(a85*a4);
  a74=(a29*a39);
  a47=(a47+a74);
  a74=(a47*a68);
  a2=(a2+a74);
  a4=(a66*a4);
  a39=(a81*a39);
  a4=(a4+a39);
  a39=(a4*a33);
  a2=(a2+a39);
  a39=(a19*a2);
  a74=(a48*a27);
  a5=(a47*a67);
  a74=(a74+a5);
  a5=(a4*a25);
  a74=(a74+a5);
  a5=(a23*a74);
  a84=(a84*a41);
  a78=(a78*a3);
  a78=(a78+a6);
  a38=(a38*a78);
  a84=(a84+a38);
  a38=(a84*a27);
  a85=(a85*a41);
  a29=(a29*a78);
  a85=(a85+a29);
  a29=(a85*a67);
  a38=(a38+a29);
  a66=(a66*a41);
  a81=(a81*a78);
  a66=(a66+a81);
  a81=(a66*a25);
  a38=(a38+a81);
  a81=(a46*a38);
  a5=(a5+a81);
  a0=(a0*a62);
  a53=(a53*a72);
  a0=(a0-a53);
  a27=(a0*a27);
  a14=(a14*a62);
  a71=(a71*a72);
  a14=(a14-a71);
  a67=(a14*a67);
  a27=(a27+a67);
  a63=(a63*a62);
  a51=(a51*a72);
  a63=(a63-a51);
  a25=(a63*a25);
  a27=(a27+a25);
  a25=(a57*a27);
  a5=(a5+a25);
  a39=(a39-a5);
  a5=(a84*a35);
  a25=(a85*a68);
  a5=(a5+a25);
  a25=(a66*a33);
  a5=(a5+a25);
  a25=(a44*a5);
  a39=(a39+a25);
  a35=(a0*a35);
  a68=(a14*a68);
  a35=(a35+a68);
  a33=(a63*a33);
  a35=(a35+a33);
  a33=(a55*a35);
  a39=(a39+a33);
  a39=(a39*a20);
  a74=(a19*a74);
  a38=(a44*a38);
  a74=(a74+a38);
  a27=(a55*a27);
  a74=(a74+a27);
  a2=(a23*a2);
  a74=(a74+a2);
  a5=(a46*a5);
  a74=(a74+a5);
  a35=(a57*a35);
  a74=(a74+a35);
  a74=(a74*a21);
  a39=(a39+a74);
  a39=(-a39);
  if (res[0]!=0) res[0][10]=a39;
  a39=(a48*a59);
  a74=(a47*a77);
  a39=(a39+a74);
  a74=(a4*a11);
  a39=(a39+a74);
  a74=(a19*a39);
  a35=(a48*a40);
  a5=(a47*a76);
  a35=(a35+a5);
  a5=(a4*a69);
  a35=(a35+a5);
  a5=(a23*a35);
  a2=(a84*a40);
  a27=(a85*a76);
  a2=(a2+a27);
  a27=(a66*a69);
  a2=(a2+a27);
  a27=(a46*a2);
  a5=(a5+a27);
  a40=(a0*a40);
  a76=(a14*a76);
  a40=(a40+a76);
  a69=(a63*a69);
  a40=(a40+a69);
  a69=(a57*a40);
  a5=(a5+a69);
  a74=(a74-a5);
  a5=(a84*a59);
  a69=(a85*a77);
  a5=(a5+a69);
  a69=(a66*a11);
  a5=(a5+a69);
  a69=(a44*a5);
  a74=(a74+a69);
  a59=(a0*a59);
  a77=(a14*a77);
  a59=(a59+a77);
  a11=(a63*a11);
  a59=(a59+a11);
  a11=(a55*a59);
  a74=(a74+a11);
  a21=(a21*a74);
  a19=(a19*a35);
  a44=(a44*a2);
  a19=(a19+a44);
  a55=(a55*a40);
  a19=(a19+a55);
  a23=(a23*a39);
  a19=(a19+a23);
  a46=(a46*a5);
  a19=(a19+a46);
  a57=(a57*a59);
  a19=(a19+a57);
  a20=(a20*a19);
  a21=(a21-a20);
  if (res[0]!=0) res[0][11]=a21;
  if (res[0]!=0) res[0][12]=a73;
  if (res[0]!=0) res[0][13]=a37;
  if (res[0]!=0) res[0][14]=a21;
  a21=(a54*a0);
  a37=(a42*a84);
  a73=(a7*a48);
  a37=(a37+a73);
  a21=(a21+a37);
  a21=(a21*a15);
  a37=(a54*a14);
  a73=(a42*a85);
  a20=(a7*a47);
  a73=(a73+a20);
  a37=(a37+a73);
  a37=(a37*a64);
  a21=(a21+a37);
  a54=(a54*a63);
  a42=(a42*a66);
  a7=(a7*a4);
  a42=(a42+a7);
  a54=(a54+a42);
  a54=(a54*a13);
  a21=(a21+a54);
  a8=(a8*a21);
  a54=(a0*a56);
  a42=(a84*a45);
  a7=(a48*a22);
  a42=(a42+a7);
  a54=(a54+a42);
  a54=(a54*a15);
  a42=(a14*a56);
  a7=(a85*a45);
  a37=(a47*a22);
  a7=(a7+a37);
  a42=(a42+a7);
  a42=(a42*a64);
  a54=(a54+a42);
  a56=(a63*a56);
  a45=(a66*a45);
  a22=(a4*a22);
  a45=(a45+a22);
  a56=(a56+a45);
  a56=(a56*a13);
  a54=(a54+a56);
  a24=(a24*a54);
  a8=(a8+a24);
  a0=(a0*a60);
  a84=(a84*a49);
  a48=(a48*a31);
  a84=(a84+a48);
  a0=(a0+a84);
  a0=(a0*a15);
  a14=(a14*a60);
  a85=(a85*a49);
  a47=(a47*a31);
  a85=(a85+a47);
  a14=(a14+a85);
  a14=(a14*a64);
  a0=(a0+a14);
  a63=(a63*a60);
  a66=(a66*a49);
  a4=(a4*a31);
  a66=(a66+a4);
  a63=(a63+a66);
  a63=(a63*a13);
  a0=(a0+a63);
  a32=(a32*a0);
  a8=(a8+a32);
  a8=(a8*a10);
  a12=(a12*a21);
  a26=(a26*a54);
  a12=(a12+a26);
  a34=(a34*a0);
  a12=(a12+a34);
  a12=(a12*a9);
  a8=(a8+a12);
  a8=(-a8);
  if (res[0]!=0) res[0][15]=a8;
  return 0;
}

CASADI_SYMBOL_EXPORT int robot_arm_cost_y_e_hess(const casadi_real** arg, casadi_real** res, casadi_int* iw, casadi_real* w, int mem){
  return casadi_f0(arg, res, iw, w, mem);
}

CASADI_SYMBOL_EXPORT int robot_arm_cost_y_e_hess_alloc_mem(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT int robot_arm_cost_y_e_hess_init_mem(int mem) {
  return 0;
}

CASADI_SYMBOL_EXPORT void robot_arm_cost_y_e_hess_free_mem(int mem) {
}

CASADI_SYMBOL_EXPORT int robot_arm_cost_y_e_hess_checkout(void) {
  return 0;
}

CASADI_SYMBOL_EXPORT void robot_arm_cost_y_e_hess_release(int mem) {
}

CASADI_SYMBOL_EXPORT void robot_arm_cost_y_e_hess_incref(void) {
}

CASADI_SYMBOL_EXPORT void robot_arm_cost_y_e_hess_decref(void) {
}

CASADI_SYMBOL_EXPORT casadi_int robot_arm_cost_y_e_hess_n_in(void) { return 6;}

CASADI_SYMBOL_EXPORT casadi_int robot_arm_cost_y_e_hess_n_out(void) { return 1;}

CASADI_SYMBOL_EXPORT casadi_real robot_arm_cost_y_e_hess_default_in(casadi_int i) {
  switch (i) {
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* robot_arm_cost_y_e_hess_name_in(casadi_int i) {
  switch (i) {
    case 0: return "i0";
    case 1: return "i1";
    case 2: return "i2";
    case 3: return "i3";
    case 4: return "i4";
    case 5: return "i5";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const char* robot_arm_cost_y_e_hess_name_out(casadi_int i) {
  switch (i) {
    case 0: return "o0";
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* robot_arm_cost_y_e_hess_sparsity_in(casadi_int i) {
  switch (i) {
    case 0: return casadi_s0;
    case 1: return casadi_s1;
    case 2: return casadi_s2;
    case 3: return casadi_s3;
    case 4: return casadi_s1;
    case 5: return casadi_s3;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT const casadi_int* robot_arm_cost_y_e_hess_sparsity_out(casadi_int i) {
  switch (i) {
    case 0: return casadi_s4;
    default: return 0;
  }
}

CASADI_SYMBOL_EXPORT int robot_arm_cost_y_e_hess_work(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6;
  if (sz_res) *sz_res = 1;
  if (sz_iw) *sz_iw = 0;
  if (sz_w) *sz_w = 0;
  return 0;
}

CASADI_SYMBOL_EXPORT int robot_arm_cost_y_e_hess_work_bytes(casadi_int *sz_arg, casadi_int* sz_res, casadi_int *sz_iw, casadi_int *sz_w) {
  if (sz_arg) *sz_arg = 6*sizeof(const casadi_real*);
  if (sz_res) *sz_res = 1*sizeof(casadi_real*);
  if (sz_iw) *sz_iw = 0*sizeof(casadi_int);
  if (sz_w) *sz_w = 0*sizeof(casadi_real);
  return 0;
}


#ifdef __cplusplus
} /* extern "C" */
#endif
