OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[1],q[0];
cx q[4],q[1];
cx q[7],q[4];
cx q[4],q[1];
cx q[1],q[0];
cx q[4],q[1];
cx q[7],q[4];
cx q[4],q[1];
t q[13];
cx q[10],q[7];
cx q[7],q[4];
cx q[4],q[1];
cx q[7],q[10];
cx q[4],q[7];
cx q[1],q[4];
cx q[0],q[1];
cx q[1],q[4];
cx q[4],q[1];
cx q[4],q[7];
cx q[7],q[4];
cx q[7],q[10];
cx q[10],q[7];
t q[1];
cx q[14],q[11];
cx q[11],q[8];
cx q[8],q[5];
cx q[5],q[3];
cx q[11],q[14];
cx q[8],q[11];
cx q[5],q[8];
cx q[3],q[5];
cx q[2],q[3];
cx q[3],q[5];
cx q[5],q[3];
cx q[5],q[8];
cx q[8],q[5];
cx q[8],q[11];
cx q[11],q[8];
cx q[11],q[14];
cx q[14],q[11];
s q[9];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[11],q[14];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[14],q[11];
cx q[13],q[14];
cx q[12],q[13];
cx q[10],q[12];
s q[7];
cx q[14],q[13];
s q[6];
cx q[15],q[12];
cx q[12],q[13];
cx q[12],q[15];
cx q[13],q[12];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[15];
cx q[15],q[12];
t q[10];
t q[4];
s q[7];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[12],q[15];
cx q[13],q[12];
cx q[14],q[13];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[14];
cx q[13],q[12];
cx q[12],q[13];
cx q[12],q[15];
cx q[15],q[12];
x q[5];
t q[0];
x q[14];
h q[12];
s q[12];
s q[12];
x q[3];
s q[13];
t q[15];
h q[14];
t q[4];
x q[6];
cx q[4],q[1];
cx q[7],q[4];
cx q[10],q[7];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[7];
cx q[7],q[4];
cx q[4],q[1];
cx q[7],q[4];
cx q[10],q[7];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[7];
cx q[7],q[4];
x q[8];
t q[15];
t q[7];
cx q[4],q[7];
t q[6];
x q[1];
cx q[5],q[3];
cx q[8],q[5];
cx q[11],q[8];
cx q[14],q[11];
cx q[13],q[14];
cx q[12],q[13];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[11],q[8];
cx q[8],q[5];
cx q[5],q[3];
cx q[8],q[5];
cx q[11],q[8];
cx q[14],q[11];
cx q[13],q[14];
cx q[12],q[13];
cx q[15],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[11],q[8];
cx q[8],q[5];
cx q[7],q[6];
cx q[10],q[7];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[7];
cx q[7],q[6];
cx q[10],q[7];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[10],q[7];
h q[5];
x q[4];
s q[13];
cx q[10],q[12];
cx q[12],q[13];
cx q[13],q[14];
cx q[14],q[11];
cx q[12],q[10];
cx q[13],q[12];
cx q[14],q[13];
cx q[11],q[14];
cx q[8],q[11];
cx q[11],q[14];
cx q[14],q[13];
cx q[13],q[12];
cx q[12],q[10];
cx q[14],q[11];
cx q[13],q[14];
cx q[12],q[13];
cx q[10],q[12];
s q[8];
cx q[11],q[8];
cx q[8],q[5];
cx q[5],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[8],q[11];
cx q[5],q[8];
cx q[3],q[5];
cx q[2],q[3];
cx q[1],q[2];
cx q[0],q[1];
cx q[1],q[2];
cx q[2],q[1];
cx q[14],q[11];
cx q[11],q[8];
cx q[8],q[5];
cx q[11],q[14];
cx q[8],q[11];
cx q[5],q[8];
cx q[3],q[5];
cx q[2],q[3];
cx q[3],q[2];
cx q[5],q[8];
cx q[3],q[5];
cx q[8],q[5];
cx q[5],q[3];
cx q[8],q[11];
cx q[5],q[8];
cx q[11],q[8];
cx q[8],q[5];
cx q[11],q[14];
cx q[8],q[11];
cx q[14],q[11];
cx q[11],q[8];
s q[8];
h q[11];
cx q[10],q[7];
h q[8];
h q[8];
t q[11];
x q[0];