OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
u2(0,3.14159265358979) q[17];
u2(0,3.14159265358979) q[12];
u2(0,3.14159265358979) q[10];
u2(0,3.14159265358979) q[9];
u2(0,3.14159265358979) q[8];
cx q[8],q[9];
u2(0,3.14159265358979) q[8];
cx q[13],q[6];
cx q[6],q[8];
u2(0,3.14159265358979) q[6];
cx q[6],q[9];
u2(0,3.14159265358979) q[9];
cx q[8],q[9];
u2(0,3.14159265358979) q[8];
u2(0,3.14159265358979) q[9];
cx q[8],q[9];
u2(0,3.14159265358979) q[8];
u2(0,3.14159265358979) q[9];
cx q[8],q[9];
cx q[9],q[19];
u2(0,3.14159265358979) q[19];
u2(0,3.14159265358979) q[6];
u2(0,3.14159265358979) q[13];
cx q[10],q[13];
u2(0,3.14159265358979) q[4];
cx q[12],q[4];
u2(0,3.14159265358979) q[4];
u2(0,3.14159265358979) q[12];
cx q[16],q[12];
u2(0,3.14159265358979) q[16];
cx q[17],q[16];
u2(0,3.14159265358979) q[16];
u2(0,3.14159265358979) q[17];
cx q[7],q[3];
cx q[14],q[7];
cx q[14],q[12];
u2(0,3.14159265358979) q[14];
u2(0,3.14159265358979) q[12];
cx q[14],q[12];
u2(0,3.14159265358979) q[14];
u2(0,3.14159265358979) q[12];
cx q[14],q[12];
u2(0,3.14159265358979) q[12];
u2(0,3.14159265358979) q[14];
u2(0,3.14159265358979) q[7];
cx q[10],q[7];
u2(0,3.14159265358979) q[7];
u2(0,3.14159265358979) q[10];
cx q[10],q[7];
u2(0,3.14159265358979) q[10];
u2(0,3.14159265358979) q[7];
cx q[10],q[7];
u2(0,3.14159265358979) q[10];
u2(0,3.14159265358979) q[7];
cx q[10],q[7];
cx q[7],q[4];
u2(0,3.14159265358979) q[7];
u2(0,3.14159265358979) q[4];
cx q[7],q[4];
u2(0,3.14159265358979) q[7];
u2(0,3.14159265358979) q[4];
cx q[7],q[4];
u2(0,3.14159265358979) q[4];
cx q[4],q[19];
u2(0,3.14159265358979) q[4];
u2(0,3.14159265358979) q[3];
u2(0,3.14159265358979) q[2];
cx q[18],q[1];
cx q[5],q[18];
cx q[18],q[6];
u2(0,3.14159265358979) q[18];
u2(0,3.14159265358979) q[6];
cx q[18],q[6];
u2(0,3.14159265358979) q[18];
u2(0,3.14159265358979) q[6];
cx q[18],q[6];
u2(0,3.14159265358979) q[18];
u2(0,3.14159265358979) q[6];
cx q[6],q[14];
u2(0,3.14159265358979) q[6];
u2(0,3.14159265358979) q[14];
cx q[6],q[14];
cx q[9],q[14];
cx q[8],q[14];
u2(0,3.14159265358979) q[14];
cx q[14],q[12];
u2(0,3.14159265358979) q[12];
u2(0,3.14159265358979) q[14];
u2(0,3.14159265358979) q[6];
cx q[18],q[6];
u2(0,3.14159265358979) q[6];
u2(0,3.14159265358979) q[18];
cx q[18],q[6];
u2(0,3.14159265358979) q[6];
u2(0,3.14159265358979) q[18];
cx q[18],q[6];
u2(0,3.14159265358979) q[18];
u2(0,3.14159265358979) q[6];
cx q[18],q[6];
u2(0,3.14159265358979) q[5];
cx q[5],q[13];
u2(0,3.14159265358979) q[13];
u2(0,3.14159265358979) q[5];
cx q[15],q[5];
u2(0,3.14159265358979) q[15];
u2(0,3.14159265358979) q[5];
cx q[15],q[5];
u2(0,3.14159265358979) q[15];
u2(0,3.14159265358979) q[5];
cx q[15],q[5];
u2(0,3.14159265358979) q[5];
cx q[11],q[0];
cx q[1],q[11];
cx q[18],q[1];
u2(0,3.14159265358979) q[18];
u2(0,3.14159265358979) q[1];
cx q[18],q[1];
u2(0,3.14159265358979) q[18];
u2(0,3.14159265358979) q[1];
cx q[18],q[1];
u2(0,3.14159265358979) q[1];
u2(0,3.14159265358979) q[18];
cx q[5],q[18];
u2(0,3.14159265358979) q[18];
u2(0,3.14159265358979) q[5];
cx q[5],q[6];
u2(0,3.14159265358979) q[5];
u2(0,3.14159265358979) q[6];
cx q[5],q[6];
u2(0,3.14159265358979) q[5];
u2(0,3.14159265358979) q[6];
cx q[5],q[6];
cx q[6],q[9];
cx q[8],q[9];
u2(0,3.14159265358979) q[8];
u2(0,3.14159265358979) q[9];
cx q[8],q[9];
u2(0,3.14159265358979) q[8];
u2(0,3.14159265358979) q[9];
cx q[8],q[9];
u2(0,3.14159265358979) q[9];
cx q[9],q[19];
u2(0,3.14159265358979) q[19];
u2(0,3.14159265358979) q[9];
cx q[11],q[10];
u2(0,3.14159265358979) q[11];
u2(0,3.14159265358979) q[10];
cx q[11],q[10];
u2(0,3.14159265358979) q[11];
u2(0,3.14159265358979) q[10];
cx q[11],q[10];
u2(0,3.14159265358979) q[10];
cx q[10],q[3];
cx q[3],q[2];
u2(0,3.14159265358979) q[2];
u2(0,3.14159265358979) q[3];
cx q[13],q[3];
u2(0,3.14159265358979) q[13];
u2(0,3.14159265358979) q[3];
cx q[13],q[3];
u2(0,3.14159265358979) q[13];
u2(0,3.14159265358979) q[3];
cx q[13],q[3];
cx q[3],q[17];
u2(0,3.14159265358979) q[10];
cx q[10],q[7];
u2(0,3.14159265358979) q[0];
cx q[1],q[0];
u2(0,3.14159265358979) q[0];
u2(0,3.14159265358979) q[1];
cx q[1],q[10];
u2(0,3.14159265358979) q[1];
u2(0,3.14159265358979) q[10];
cx q[1],q[10];
u2(0,3.14159265358979) q[1];
u2(0,3.14159265358979) q[10];
cx q[1],q[10];
cx q[10],q[3];
