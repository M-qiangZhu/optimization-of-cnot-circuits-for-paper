OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
creg c[16];
cx q[12],q[7];
cx q[13],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[12],q[7];
cx q[12],q[7];
cx q[13],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[5];
cx q[12],q[5];
cx q[15],q[7];
cx q[15],q[7];
cx q[15],q[14];
cx q[14],q[7];
cx q[14],q[7];
cx q[15],q[14];
cx q[14],q[7];
cx q[14],q[7];
cx q[14],q[13];
cx q[13],q[7];
cx q[13],q[7];
cx q[15],q[13];
cx q[13],q[7];
cx q[13],q[7];
cx q[14],q[13];
cx q[13],q[7];
cx q[13],q[7];
cx q[15],q[13];
cx q[13],q[7];
cx q[13],q[7];
cx q[13],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[12],q[5];
cx q[12],q[5];
cx q[13],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[3];
cx q[12],q[3];
cx q[15],q[5];
cx q[15],q[5];
cx q[15],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[15],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[14],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[15],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[14],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[15],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[13],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[12],q[3];
cx q[12],q[3];
cx q[13],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[2];
cx q[12],q[2];
cx q[15],q[3];
cx q[15],q[3];
cx q[15],q[14];
cx q[14],q[3];
cx q[14],q[3];
cx q[15],q[14];
cx q[14],q[3];
cx q[14],q[3];
cx q[14],q[13];
cx q[13],q[3];
cx q[13],q[3];
cx q[15],q[13];
cx q[13],q[3];
cx q[13],q[3];
cx q[14],q[13];
cx q[13],q[3];
cx q[13],q[3];
cx q[15],q[13];
cx q[13],q[3];
cx q[13],q[3];
cx q[13],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[12],q[2];
cx q[12],q[2];
cx q[13],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[13],q[12];
cx q[15],q[2];
cx q[15],q[2];
cx q[15],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[15],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[14],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[15],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[14],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[15],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[12],q[1];
cx q[12],q[1];
cx q[13],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[12],q[1];
cx q[12],q[1];
cx q[13],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[13],q[12];
cx q[15],q[1];
cx q[15],q[1];
cx q[15],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[15],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[14],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[15],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[14],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[15],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[15],q[14];
cx q[15],q[14];
cx q[14],q[13];
cx q[15],q[13];
cx q[14],q[13];
cx q[15],q[13];
cx q[13],q[6];
cx q[14],q[6];
cx q[13],q[6];
cx q[15],q[6];
cx q[13],q[6];
cx q[14],q[6];
cx q[13],q[6];
cx q[15],q[6];
cx q[15],q[14];
cx q[15],q[14];
cx q[14],q[13];
cx q[15],q[13];
cx q[14],q[13];
cx q[15],q[13];
cx q[13],q[3];
cx q[14],q[3];
cx q[13],q[3];
cx q[15],q[3];
cx q[13],q[3];
cx q[14],q[3];
cx q[13],q[3];
cx q[15],q[3];
cx q[15],q[14];
cx q[15],q[14];
cx q[14],q[13];
cx q[15],q[13];
cx q[14],q[13];
cx q[15],q[13];
cx q[13],q[2];
cx q[14],q[2];
cx q[13],q[2];
cx q[15],q[2];
cx q[13],q[2];
cx q[14],q[2];
cx q[13],q[2];
cx q[15],q[2];
cx q[12],q[7];
cx q[12],q[7];
cx q[13],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[12],q[7];
cx q[12],q[7];
cx q[13],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[4];
cx q[12],q[4];
cx q[15],q[7];
cx q[15],q[7];
cx q[15],q[14];
cx q[14],q[7];
cx q[14],q[7];
cx q[15],q[14];
cx q[14],q[7];
cx q[14],q[7];
cx q[14],q[13];
cx q[13],q[7];
cx q[13],q[7];
cx q[15],q[13];
cx q[13],q[7];
cx q[13],q[7];
cx q[14],q[13];
cx q[13],q[7];
cx q[13],q[7];
cx q[15],q[13];
cx q[13],q[7];
cx q[13],q[7];
cx q[13],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[12],q[4];
cx q[12],q[4];
cx q[13],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[3];
cx q[12],q[3];
cx q[15],q[4];
cx q[15],q[4];
cx q[15],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[15],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[14],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[15],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[14],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[15],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[13],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[12],q[3];
cx q[12],q[3];
cx q[13],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[13],q[12];
cx q[15],q[3];
cx q[15],q[3];
cx q[15],q[14];
cx q[14],q[3];
cx q[14],q[3];
cx q[15],q[14];
cx q[14],q[3];
cx q[14],q[3];
cx q[14],q[13];
cx q[13],q[3];
cx q[13],q[3];
cx q[15],q[13];
cx q[13],q[3];
cx q[13],q[3];
cx q[14],q[13];
cx q[13],q[3];
cx q[13],q[3];
cx q[15],q[13];
cx q[13],q[3];
cx q[13],q[3];
cx q[10],q[6];
cx q[10],q[6];
cx q[12],q[2];
cx q[12],q[2];
cx q[13],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[12],q[2];
cx q[12],q[2];
cx q[13],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[13],q[12];
cx q[12],q[1];
cx q[12],q[1];
cx q[15],q[2];
cx q[15],q[2];
cx q[15],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[15],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[14],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[15],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[14],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[15],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[13],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[12],q[1];
cx q[12],q[1];
cx q[13],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[13],q[12];
cx q[10],q[12];
cx q[12],q[6];
cx q[15],q[1];
cx q[15],q[1];
cx q[15],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[15],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[14],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[15],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[14],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[15],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[12],q[6];
cx q[10],q[12];
cx q[12],q[6];
cx q[12],q[6];
cx q[12],q[13];
cx q[13],q[6];
cx q[13],q[6];
cx q[10],q[13];
cx q[13],q[6];
cx q[13],q[6];
cx q[12],q[13];
cx q[13],q[6];
cx q[13],q[6];
cx q[10],q[13];
cx q[13],q[6];
cx q[13],q[6];
cx q[13],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[10],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[12],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[10],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[13],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[10],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[12],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[10],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[14],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[10],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[12],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[10],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[13],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[10],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[12],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[10],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[14],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[10],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[12],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[10],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[13],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[10],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[12],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[10],q[15];
cx q[10],q[5];
cx q[10],q[5];
cx q[10],q[12];
cx q[12],q[5];
cx q[12],q[5];
cx q[10],q[12];
cx q[12],q[5];
cx q[12],q[5];
cx q[12],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[10],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[12],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[10],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[13],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[10],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[12],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[10],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[13],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[10],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[12],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[10],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[15],q[6];
cx q[15],q[6];
cx q[14],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[10],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[12],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[10],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[13],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[10],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[12],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[10],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[14],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[10],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[12],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[10],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[13],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[10],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[12],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[10],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[10],q[4];
cx q[10],q[4];
cx q[10],q[12];
cx q[12],q[4];
cx q[12],q[4];
cx q[10],q[12];
cx q[12],q[4];
cx q[12],q[4];
cx q[12],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[10],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[12],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[10],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[13],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[10],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[12],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[10],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[13],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[10],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[12],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[10],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[14],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[10],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[12],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[10],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[13],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[10],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[12],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[10],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[14],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[10],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[12],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[10],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[13],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[10],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[12],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[10],q[15];
cx q[10],q[2];
cx q[10],q[2];
cx q[10],q[12];
cx q[12],q[2];
cx q[12],q[2];
cx q[10],q[12];
cx q[12],q[2];
cx q[12],q[2];
cx q[12],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[10],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[12],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[10],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[13],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[10],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[12],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[10],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[13],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[10],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[12],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[10],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[15],q[4];
cx q[15],q[4];
cx q[14],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[10],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[12],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[10],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[13],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[10],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[12],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[10],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[14],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[10],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[12],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[10],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[13],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[10],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[12],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[10],q[15];
cx q[10],q[1];
cx q[10],q[1];
cx q[10],q[12];
cx q[12],q[1];
cx q[12],q[1];
cx q[10],q[12];
cx q[12],q[1];
cx q[12],q[1];
cx q[12],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[10],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[12],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[10],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[13],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[10],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[12],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[10],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[13],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[10],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[12],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[10],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[15],q[2];
cx q[15],q[2];
cx q[14],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[10],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[12],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[10],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[13],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[10],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[12],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[10],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[14],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[10],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[12],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[10],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[13],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[10],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[12],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[10],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[15],q[14];
cx q[15],q[14];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[12],q[6];
cx q[14],q[6];
cx q[12],q[6];
cx q[15],q[6];
cx q[12],q[6];
cx q[14],q[6];
cx q[12],q[6];
cx q[15],q[6];
cx q[15],q[14];
cx q[15],q[14];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[12],q[5];
cx q[14],q[5];
cx q[12],q[5];
cx q[15],q[5];
cx q[12],q[5];
cx q[14],q[5];
cx q[12],q[5];
cx q[15],q[5];
cx q[15],q[14];
cx q[15],q[14];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[12],q[4];
cx q[14],q[4];
cx q[12],q[4];
cx q[15],q[4];
cx q[12],q[4];
cx q[14],q[4];
cx q[12],q[4];
cx q[15],q[4];
cx q[15],q[14];
cx q[15],q[14];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[12],q[2];
cx q[14],q[2];
cx q[12],q[2];
cx q[15],q[2];
cx q[12],q[2];
cx q[14],q[2];
cx q[12],q[2];
cx q[15],q[2];
cx q[15],q[14];
cx q[15],q[14];
cx q[14],q[12];
cx q[15],q[12];
cx q[14],q[12];
cx q[15],q[12];
cx q[12],q[1];
cx q[14],q[1];
cx q[12],q[1];
cx q[15],q[1];
cx q[12],q[1];
cx q[14],q[1];
cx q[12],q[1];
cx q[15],q[1];
cx q[15],q[13];
cx q[15],q[13];
cx q[13],q[12];
cx q[15],q[12];
cx q[13],q[12];
cx q[15],q[12];
cx q[12],q[5];
cx q[13],q[5];
cx q[12],q[5];
cx q[15],q[5];
cx q[12],q[5];
cx q[13],q[5];
cx q[13],q[6];
cx q[12],q[5];
cx q[15],q[5];
cx q[15],q[6];
cx q[13],q[6];
cx q[15],q[6];
cx q[15],q[13];
cx q[15],q[13];
cx q[13],q[3];
cx q[15],q[3];
cx q[13],q[3];
cx q[15],q[3];
cx q[15],q[13];
cx q[15],q[13];
cx q[13],q[2];
cx q[15],q[2];
cx q[13],q[2];
cx q[15],q[2];
cx q[15],q[13];
cx q[15],q[13];
cx q[13],q[1];
cx q[15],q[1];
cx q[13],q[1];
cx q[15],q[1];
cx q[15],q[13];
cx q[15],q[13];
cx q[14],q[13];
cx q[14],q[13];
cx q[13],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[14],q[12];
cx q[12],q[6];
cx q[13],q[6];
cx q[12],q[6];
cx q[14],q[6];
cx q[12],q[6];
cx q[13],q[6];
cx q[12],q[6];
cx q[14],q[6];
cx q[14],q[13];
cx q[14],q[13];
cx q[13],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[14],q[12];
cx q[12],q[5];
cx q[13],q[5];
cx q[12],q[5];
cx q[14],q[5];
cx q[12],q[5];
cx q[13],q[5];
cx q[12],q[5];
cx q[14],q[5];
cx q[14],q[13];
cx q[14],q[13];
cx q[13],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[14],q[12];
cx q[12],q[3];
cx q[13],q[3];
cx q[12],q[3];
cx q[14],q[3];
cx q[12],q[3];
cx q[13],q[3];
cx q[12],q[3];
cx q[14],q[3];
cx q[14],q[13];
cx q[14],q[13];
cx q[13],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[14],q[12];
cx q[12],q[2];
cx q[13],q[2];
cx q[12],q[2];
cx q[14],q[2];
cx q[12],q[2];
cx q[13],q[2];
cx q[12],q[2];
cx q[14],q[2];
cx q[14],q[13];
cx q[14],q[13];
cx q[13],q[12];
cx q[14],q[12];
cx q[13],q[12];
cx q[14],q[12];
cx q[12],q[1];
cx q[13],q[1];
cx q[12],q[1];
cx q[14],q[1];
cx q[12],q[1];
cx q[13],q[1];
cx q[12],q[1];
cx q[14],q[1];
cx q[11],q[5];
cx q[11],q[5];
cx q[15],q[13];
cx q[15],q[13];
cx q[13],q[12];
cx q[15],q[12];
cx q[13],q[12];
cx q[15],q[12];
cx q[12],q[4];
cx q[13],q[4];
cx q[12],q[4];
cx q[15],q[4];
cx q[12],q[4];
cx q[13],q[4];
cx q[12],q[4];
cx q[11],q[12];
cx q[12],q[5];
cx q[15],q[4];
cx q[12],q[5];
cx q[11],q[12];
cx q[12],q[5];
cx q[12],q[5];
cx q[12],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[11],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[12],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[11],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[13],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[11],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[12],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[11],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[13],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[11],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[12],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[11],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[14],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[11],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[12],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[11],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[13],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[11],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[12],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[11],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[14],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[11],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[12],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[11],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[13],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[11],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[12],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[11],q[15];
cx q[11],q[4];
cx q[11],q[4];
cx q[11],q[12];
cx q[12],q[4];
cx q[12],q[4];
cx q[11],q[12];
cx q[12],q[4];
cx q[12],q[4];
cx q[12],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[11],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[12],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[11],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[13],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[11],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[12],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[11],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[13],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[11],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[12],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[11],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[15],q[5];
cx q[15],q[5];
cx q[14],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[12],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[13],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[12],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[14],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[12],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[13],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[12],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[9],q[6];
cx q[9],q[6];
cx q[11],q[1];
cx q[11],q[1];
cx q[11],q[12];
cx q[12],q[1];
cx q[12],q[1];
cx q[11],q[12];
cx q[12],q[1];
cx q[12],q[1];
cx q[12],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[11],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[12],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[11],q[13];
cx q[13],q[1];
cx q[13],q[1];
cx q[13],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[11],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[12],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[11],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[13],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[11],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[12],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[11],q[14];
cx q[14],q[1];
cx q[14],q[1];
cx q[14],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[11],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[12],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[11],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[13],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[11],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[12],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[11],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[14],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[11],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[12],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[11],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[13],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[11],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[12],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[11],q[15];
cx q[15],q[1];
cx q[15],q[1];
cx q[9],q[12];
cx q[12],q[6];
cx q[12],q[6];
cx q[9],q[12];
cx q[12],q[6];
cx q[12],q[6];
cx q[12],q[13];
cx q[13],q[6];
cx q[13],q[6];
cx q[9],q[13];
cx q[13],q[6];
cx q[13],q[6];
cx q[12],q[13];
cx q[13],q[6];
cx q[13],q[6];
cx q[9],q[13];
cx q[13],q[6];
cx q[13],q[6];
cx q[13],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[9],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[12],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[9],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[13],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[9],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[12],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[9],q[14];
cx q[14],q[6];
cx q[14],q[6];
cx q[14],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[9],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[12],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[9],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[13],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[9],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[12],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[9],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[14],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[9],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[12],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[9],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[13],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[9],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[12],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[9],q[15];
cx q[15],q[6];
cx q[15],q[6];
cx q[9],q[5];
cx q[9],q[5];
cx q[9],q[12];
cx q[12],q[5];
cx q[12],q[5];
cx q[9],q[12];
cx q[12],q[5];
cx q[12],q[5];
cx q[12],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[9],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[12],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[9],q[13];
cx q[13],q[5];
cx q[13],q[5];
cx q[13],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[9],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[12],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[9],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[13],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[9],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[12],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[9],q[14];
cx q[14],q[5];
cx q[14],q[5];
cx q[14],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[9],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[12],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[9],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[13],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[9],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[12],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[9],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[14],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[9],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[12],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[9],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[13],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[9],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[12],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[9],q[15];
cx q[15],q[5];
cx q[15],q[5];
cx q[9],q[2];
cx q[9],q[2];
cx q[9],q[12];
cx q[12],q[2];
cx q[12],q[2];
cx q[9],q[12];
cx q[12],q[2];
cx q[12],q[2];
cx q[12],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[9],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[12],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[9],q[13];
cx q[13],q[2];
cx q[13],q[2];
cx q[13],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[9],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[12],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[9],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[13],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[9],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[12],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[9],q[14];
cx q[14],q[2];
cx q[14],q[2];
cx q[14],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[9],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[12],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[9],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[13],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[9],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[12],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[9],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[14],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[9],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[12],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[9],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[13],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[9],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[12],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[9],q[15];
cx q[15],q[2];
cx q[15],q[2];
cx q[8],q[4];
cx q[8],q[4];
cx q[13],q[8];
cx q[15],q[8];
cx q[14],q[8];
cx q[15],q[8];
cx q[14],q[8];
cx q[13],q[8];
cx q[8],q[4];
cx q[8],q[4];
cx q[13],q[8];
cx q[14],q[8];
cx q[15],q[8];
cx q[14],q[8];
cx q[15],q[8];
cx q[15],q[4];
cx q[15],q[4];
cx q[15],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[15],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[13],q[8];
cx q[14],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[15],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[14],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[15],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[8],q[4];
cx q[8],q[4];
cx q[8],q[11];
cx q[11],q[4];
cx q[11],q[4];
cx q[8],q[11];
cx q[11],q[4];
cx q[11],q[4];
cx q[11],q[12];
cx q[12],q[4];
cx q[12],q[4];
cx q[8],q[12];
cx q[12],q[4];
cx q[12],q[4];
cx q[11],q[12];
cx q[12],q[4];
cx q[12],q[4];
cx q[8],q[12];
cx q[12],q[4];
cx q[12],q[4];
cx q[12],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[8],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[11],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[8],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[12],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[8],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[11],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[8],q[13];
cx q[13],q[4];
cx q[13],q[4];
cx q[13],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[8],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[11],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[8],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[12],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[8],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[11],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[8],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[13],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[8],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[11],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[8],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[12],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[8],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[11],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[8],q[14];
cx q[14],q[4];
cx q[14],q[4];
cx q[14],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[12],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[13],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[12],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[14],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[12],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[13],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[12],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[11],q[15];
cx q[15],q[4];
cx q[15],q[4];
cx q[8],q[15];
cx q[15],q[4];
cx q[15],q[4];
