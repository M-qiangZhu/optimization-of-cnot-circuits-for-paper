OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[4],q[14];
cx q[2],q[11];
cx q[9],q[10];
x q[13];
x q[3];
x q[5];
cx q[11],q[10];
cx q[5],q[13];
s q[0];
cx q[13],q[11];
cx q[3],q[15];
s q[15];
cx q[7],q[1];
t q[15];
t q[14];
cx q[1],q[6];
cx q[11],q[15];
cx q[8],q[12];
t q[6];
s q[3];
x q[4];
cx q[4],q[12];
s q[1];
cx q[4],q[5];
cx q[13],q[15];
cx q[11],q[6];
cx q[4],q[7];
x q[8];
cx q[14],q[15];
cx q[4],q[10];
cx q[15],q[10];
cx q[3],q[11];
cx q[14],q[10];
h q[12];
cx q[0],q[4];
s q[6];
h q[15];
t q[9];
h q[1];
cx q[5],q[10];
s q[9];
x q[2];
cx q[13],q[10];
cx q[9],q[1];
cx q[9],q[10];
cx q[5],q[6];
cx q[6],q[8];
t q[2];
cx q[8],q[6];
h q[14];
cx q[3],q[14];
cx q[8],q[4];
cx q[7],q[15];
x q[3];
cx q[2],q[4];
cx q[5],q[9];
s q[9];
cx q[2],q[0];
h q[15];
cx q[1],q[0];
h q[5];
cx q[15],q[6];
cx q[11],q[7];
h q[10];
cx q[10],q[9];
s q[0];
cx q[15],q[12];
cx q[3],q[2];
s q[14];
cx q[4],q[6];
cx q[6],q[11];
cx q[4],q[5];
s q[14];
s q[12];
t q[15];
s q[0];
cx q[10],q[6];
cx q[0],q[14];
s q[7];
cx q[2],q[8];
t q[10];
x q[9];
s q[8];
x q[15];
h q[3];
cx q[12],q[0];
x q[8];
s q[8];
cx q[4],q[9];
cx q[4],q[1];
cx q[4],q[2];
cx q[13],q[14];
cx q[14],q[3];
x q[8];
s q[1];
cx q[13],q[4];
cx q[9],q[2];
x q[8];
x q[10];
cx q[8],q[1];
h q[4];
s q[12];
s q[14];
s q[13];
x q[12];
cx q[12],q[15];
cx q[14],q[15];
cx q[0],q[8];
t q[13];
cx q[6],q[11];
s q[7];
cx q[14],q[2];
x q[9];
cx q[13],q[3];
t q[13];
s q[7];
cx q[12],q[10];
cx q[2],q[15];
x q[9];
cx q[14],q[10];
cx q[14],q[1];
cx q[0],q[14];
cx q[11],q[14];
cx q[7],q[14];
t q[14];
cx q[0],q[11];
cx q[12],q[5];
cx q[15],q[12];
cx q[0],q[5];
cx q[2],q[12];
cx q[12],q[11];
x q[1];
cx q[1],q[3];
t q[4];
h q[0];
s q[14];
cx q[14],q[9];
cx q[15],q[13];
x q[15];
s q[0];
s q[13];
cx q[15],q[11];
cx q[3],q[11];
cx q[9],q[2];
cx q[14],q[1];
cx q[14],q[13];
cx q[6],q[14];
x q[0];
h q[10];
cx q[14],q[2];
cx q[10],q[6];
h q[0];
s q[14];
t q[5];
h q[2];
cx q[14],q[9];
x q[5];
cx q[7],q[14];
cx q[13],q[5];
t q[9];
x q[1];
cx q[12],q[1];
cx q[5],q[14];
x q[5];
cx q[8],q[12];
x q[7];
cx q[13],q[6];
x q[9];
t q[9];
h q[9];
x q[6];
cx q[13],q[15];
t q[7];
t q[4];
cx q[12],q[4];
cx q[9],q[10];
cx q[10],q[14];
x q[13];
cx q[11],q[15];
t q[15];
h q[2];
cx q[6],q[15];
cx q[11],q[9];
x q[8];
x q[4];
cx q[3],q[12];
cx q[4],q[11];
cx q[11],q[4];
s q[9];
cx q[15],q[14];
x q[11];
s q[4];
h q[0];
cx q[2],q[4];
cx q[1],q[5];
cx q[9],q[1];
cx q[1],q[13];
h q[8];
cx q[9],q[13];
cx q[1],q[6];
cx q[3],q[1];
s q[11];
s q[4];
t q[4];
t q[6];
cx q[9],q[6];
t q[13];
t q[8];
s q[5];
x q[15];
h q[13];
cx q[5],q[4];
h q[2];
cx q[11],q[12];
t q[3];
x q[8];
cx q[11],q[1];
cx q[8],q[3];
s q[1];
h q[0];
cx q[9],q[6];
cx q[6],q[15];
h q[10];
cx q[2],q[8];
h q[4];
x q[7];
x q[13];
cx q[15],q[9];
h q[2];
s q[8];
h q[0];
cx q[0],q[2];
cx q[10],q[9];
cx q[0],q[1];
cx q[15],q[8];
cx q[8],q[3];
cx q[5],q[9];
cx q[2],q[5];
x q[9];
cx q[0],q[15];
x q[5];
cx q[12],q[5];
cx q[2],q[14];
cx q[0],q[9];
t q[2];
t q[10];
cx q[5],q[3];
cx q[2],q[14];
cx q[9],q[10];
cx q[14],q[11];
h q[15];
cx q[15],q[6];
s q[7];
cx q[9],q[11];
h q[4];
cx q[3],q[2];
x q[1];
cx q[5],q[9];
s q[0];
cx q[11],q[7];
s q[14];
s q[1];
h q[8];
x q[2];
h q[6];
s q[13];
t q[0];
s q[4];
cx q[11],q[6];
cx q[0],q[11];
h q[10];
cx q[6],q[1];
cx q[10],q[15];
x q[6];
h q[13];
h q[0];
x q[14];
t q[8];
cx q[0],q[13];
t q[0];
cx q[9],q[5];
h q[11];
t q[4];
cx q[1],q[11];
cx q[10],q[3];
h q[2];
cx q[13],q[4];
cx q[14],q[2];
cx q[7],q[9];
cx q[13],q[6];
h q[3];
cx q[13],q[3];
s q[12];
x q[0];
t q[10];
cx q[11],q[3];
x q[15];
s q[4];
cx q[7],q[13];
cx q[10],q[2];
t q[11];
cx q[5],q[3];
cx q[7],q[0];
x q[4];
cx q[1],q[10];
h q[8];
s q[12];
t q[6];
t q[5];
h q[7];
cx q[1],q[6];
x q[2];
s q[1];
cx q[8],q[7];
t q[12];
s q[13];
h q[12];
x q[5];
cx q[13],q[4];
cx q[7],q[8];
cx q[2],q[8];
cx q[14],q[11];
s q[7];
cx q[5],q[2];
s q[1];
cx q[13],q[0];
cx q[12],q[4];
cx q[5],q[8];
s q[9];
t q[13];
cx q[1],q[7];
cx q[3],q[8];
cx q[9],q[12];
cx q[9],q[1];
h q[1];
cx q[6],q[11];
h q[15];
cx q[11],q[1];
x q[14];
cx q[9],q[14];
x q[10];
cx q[2],q[5];
s q[11];
cx q[7],q[14];
cx q[7],q[8];
cx q[15],q[5];
t q[1];
t q[1];
cx q[6],q[2];
cx q[11],q[1];
x q[0];
t q[5];
cx q[13],q[4];
cx q[4],q[15];
t q[9];
cx q[1],q[0];
x q[6];
cx q[14],q[6];
cx q[0],q[9];
cx q[9],q[14];
cx q[1],q[2];
x q[5];
cx q[1],q[5];
cx q[1],q[10];
s q[2];
x q[6];
x q[13];
h q[2];
cx q[15],q[11];
x q[7];
cx q[7],q[10];
t q[8];
cx q[0],q[6];
cx q[13],q[9];
x q[2];
s q[4];
cx q[7],q[3];
h q[4];
cx q[6],q[9];
cx q[12],q[6];
cx q[2],q[10];
h q[4];
x q[14];
x q[13];
s q[6];
h q[8];
x q[9];
h q[0];
cx q[15],q[3];
cx q[13],q[2];
cx q[7],q[1];
cx q[5],q[6];
h q[13];
t q[3];
x q[0];
x q[5];
x q[11];
cx q[14],q[7];
x q[10];
cx q[14],q[3];
x q[12];
x q[13];
h q[2];
s q[7];
x q[12];
h q[14];
cx q[4],q[13];
cx q[10],q[15];
x q[6];
cx q[1],q[12];
s q[0];
cx q[15],q[7];
cx q[5],q[7];
x q[13];
s q[6];
cx q[0],q[2];
t q[5];
s q[3];
x q[9];
cx q[14],q[8];
cx q[6],q[8];
x q[4];
s q[5];
cx q[8],q[14];
x q[12];
s q[5];
s q[4];
cx q[9],q[6];
cx q[14],q[0];
cx q[9],q[4];
cx q[2],q[5];
t q[15];
cx q[5],q[10];
cx q[4],q[6];
cx q[4],q[8];
cx q[12],q[5];
t q[6];
cx q[12],q[5];
x q[2];
x q[7];
cx q[14],q[5];
s q[10];
cx q[15],q[7];
s q[9];
h q[15];
cx q[2],q[11];
s q[13];
cx q[14],q[6];
h q[2];
cx q[1],q[14];
x q[15];
cx q[9],q[1];
cx q[0],q[10];
t q[4];
cx q[9],q[13];
cx q[14],q[13];
cx q[13],q[2];
s q[14];
h q[7];
cx q[4],q[8];
t q[7];
cx q[5],q[0];
cx q[0],q[10];
h q[4];
cx q[0],q[5];
cx q[10],q[11];
t q[4];
s q[0];
cx q[11],q[10];
t q[8];
s q[5];
cx q[6],q[14];
t q[6];
cx q[9],q[7];
h q[6];
cx q[2],q[5];
cx q[0],q[2];
s q[10];
t q[1];
t q[2];
cx q[9],q[8];
cx q[6],q[3];
s q[9];
t q[12];
s q[9];
cx q[10],q[0];
h q[6];
s q[6];
cx q[7],q[4];
x q[6];
cx q[9],q[10];
s q[8];
cx q[8],q[15];
cx q[14],q[13];
cx q[15],q[11];
cx q[14],q[4];
x q[1];
x q[11];
h q[6];
t q[10];
cx q[9],q[6];
cx q[2],q[15];
s q[1];
h q[1];
cx q[8],q[9];
cx q[2],q[11];
s q[5];
cx q[10],q[9];
cx q[14],q[10];
cx q[0],q[4];
x q[2];
x q[11];
cx q[5],q[4];
cx q[5],q[1];
t q[11];
cx q[7],q[10];
cx q[12],q[0];
x q[11];
cx q[15],q[8];
cx q[9],q[1];
cx q[6],q[0];
h q[9];
x q[0];
x q[0];
h q[11];
t q[15];
cx q[12],q[2];
t q[4];
cx q[10],q[8];
s q[3];
x q[9];
x q[3];
cx q[8],q[7];
s q[4];
t q[6];
t q[1];
cx q[3],q[14];
cx q[14],q[11];
cx q[10],q[15];
cx q[4],q[8];
cx q[5],q[12];
x q[9];
t q[14];
t q[12];
cx q[11],q[6];
cx q[12],q[13];
cx q[13],q[14];
cx q[7],q[14];
h q[4];
s q[14];
s q[15];
h q[2];
cx q[0],q[5];
t q[13];
t q[13];
cx q[10],q[14];
cx q[7],q[15];
h q[3];
x q[7];
cx q[0],q[7];
cx q[8],q[3];
cx q[6],q[1];
cx q[9],q[6];
cx q[5],q[10];
cx q[13],q[10];
x q[8];
h q[12];
t q[12];
t q[13];
t q[1];
cx q[9],q[1];
t q[15];
cx q[15],q[2];
cx q[0],q[10];
t q[15];
s q[9];
s q[7];
cx q[11],q[7];
h q[8];
x q[11];
cx q[2],q[6];
h q[11];
cx q[12],q[2];
cx q[2],q[12];
cx q[8],q[3];
s q[11];
s q[8];
cx q[10],q[11];
cx q[15],q[10];
h q[7];
s q[14];
cx q[15],q[6];
x q[11];
cx q[2],q[12];
cx q[2],q[3];
cx q[4],q[6];
t q[0];
cx q[14],q[15];
cx q[1],q[9];
cx q[13],q[11];
s q[1];
t q[12];
t q[8];
cx q[3],q[0];
cx q[0],q[5];
cx q[8],q[0];
cx q[7],q[15];
x q[8];
cx q[13],q[6];
s q[13];
x q[10];
h q[2];
x q[7];
cx q[1],q[9];
cx q[5],q[11];
x q[8];
h q[4];
cx q[5],q[0];
s q[13];
cx q[12],q[14];
cx q[9],q[1];
t q[9];
x q[12];
cx q[11],q[12];
cx q[12],q[4];
cx q[13],q[12];
cx q[11],q[10];
cx q[12],q[13];
x q[1];
cx q[6],q[15];
cx q[11],q[5];
h q[7];
cx q[15],q[8];
t q[1];
s q[9];
x q[14];
s q[0];
cx q[11],q[14];
h q[3];
s q[9];
cx q[7],q[5];
x q[10];
cx q[15],q[3];
x q[4];
cx q[15],q[13];
cx q[2],q[11];
cx q[10],q[6];
cx q[2],q[12];
x q[6];
t q[0];
cx q[1],q[7];
cx q[4],q[3];
cx q[11],q[6];
cx q[1],q[6];
s q[0];
t q[2];
s q[2];
h q[15];
t q[13];
s q[6];
s q[10];
t q[0];
cx q[9],q[2];
t q[12];
s q[14];
cx q[15],q[11];
h q[2];
t q[8];
cx q[4],q[9];
cx q[8],q[0];
cx q[7],q[6];
cx q[15],q[14];
s q[9];
cx q[6],q[14];
cx q[2],q[5];
cx q[2],q[11];
x q[10];
cx q[1],q[3];
t q[5];
x q[12];
s q[4];
cx q[7],q[12];
cx q[8],q[11];
h q[3];
t q[7];
x q[15];
h q[10];
t q[12];
h q[6];
cx q[3],q[7];
x q[3];
cx q[4],q[0];
cx q[6],q[9];
cx q[3],q[8];
cx q[15],q[2];
x q[1];
cx q[14],q[13];
s q[6];
cx q[5],q[1];
x q[7];
t q[14];
x q[7];
s q[3];
cx q[6],q[0];
t q[4];
cx q[14],q[7];
h q[7];
cx q[6],q[5];
cx q[6],q[8];
cx q[15],q[10];
h q[12];
cx q[8],q[12];
h q[14];
cx q[8],q[1];
s q[12];
s q[15];
h q[7];
cx q[0],q[1];
cx q[4],q[14];
cx q[9],q[5];
cx q[13],q[14];
h q[8];
s q[9];
cx q[9],q[1];
h q[9];
cx q[7],q[1];
cx q[12],q[13];
cx q[3],q[11];
cx q[2],q[9];
h q[1];
cx q[7],q[0];
s q[14];
t q[2];
cx q[10],q[15];
cx q[7],q[14];
cx q[9],q[2];
x q[5];
t q[6];
cx q[13],q[7];
s q[15];
cx q[11],q[14];
h q[0];
cx q[13],q[10];
cx q[2],q[4];
cx q[11],q[10];
cx q[9],q[7];
x q[4];
h q[9];
h q[11];
cx q[0],q[3];
cx q[11],q[0];
cx q[11],q[9];
x q[3];
cx q[1],q[12];
cx q[6],q[1];
cx q[6],q[13];
x q[11];
h q[5];
x q[13];
t q[1];
cx q[14],q[11];
t q[4];
cx q[10],q[15];
s q[7];
cx q[11],q[2];
s q[14];
cx q[6],q[1];
x q[8];
cx q[11],q[9];
cx q[3],q[2];
cx q[4],q[14];
x q[0];
x q[1];
t q[7];
cx q[3],q[4];
x q[7];
cx q[5],q[15];
t q[6];
cx q[15],q[1];
x q[10];
t q[8];
cx q[10],q[4];
cx q[8],q[0];
cx q[14],q[7];
cx q[4],q[0];
cx q[9],q[8];
s q[0];
x q[14];
x q[10];
x q[2];
s q[15];
cx q[3],q[7];
cx q[8],q[11];
t q[0];
t q[13];
cx q[9],q[10];
x q[14];
cx q[12],q[3];
x q[6];
cx q[12],q[7];
h q[10];
cx q[2],q[9];
cx q[9],q[11];
cx q[4],q[14];
h q[14];
t q[1];
h q[0];
cx q[3],q[6];
cx q[14],q[8];
cx q[12],q[1];
cx q[8],q[15];
h q[3];
h q[2];
s q[4];
cx q[9],q[5];
cx q[11],q[2];
cx q[2],q[7];
x q[5];
cx q[2],q[6];
cx q[8],q[12];
x q[9];
s q[6];
h q[12];
h q[5];
cx q[0],q[7];
cx q[2],q[8];
s q[5];
h q[9];
cx q[10],q[1];
h q[14];
cx q[5],q[6];
cx q[14],q[8];
x q[8];
x q[6];
h q[14];
x q[9];
cx q[3],q[5];
cx q[9],q[8];
cx q[5],q[6];
h q[14];
t q[12];
cx q[8],q[5];
cx q[3],q[2];
cx q[12],q[13];
cx q[13],q[8];
h q[1];
h q[10];
cx q[9],q[14];
cx q[9],q[5];
h q[6];
h q[11];
t q[8];
h q[15];
cx q[11],q[10];
h q[0];
h q[12];
s q[15];
cx q[7],q[14];
cx q[11],q[10];
cx q[15],q[6];
h q[12];
t q[0];
cx q[5],q[1];
s q[11];
h q[0];
cx q[15],q[7];
x q[5];
x q[13];
h q[2];
s q[15];
cx q[8],q[2];
h q[12];
cx q[0],q[14];
cx q[15],q[5];
cx q[6],q[7];
cx q[15],q[7];
cx q[12],q[6];
x q[5];
cx q[11],q[14];
t q[3];
t q[4];
cx q[6],q[8];
cx q[0],q[8];
t q[1];
t q[8];
h q[0];
cx q[8],q[1];
t q[4];
x q[11];
cx q[12],q[9];
cx q[6],q[11];
cx q[1],q[10];
cx q[4],q[2];
cx q[9],q[13];
t q[1];
t q[9];
cx q[4],q[11];
cx q[2],q[10];
cx q[2],q[14];
cx q[11],q[13];
x q[9];
cx q[11],q[8];
x q[3];
cx q[13],q[14];
cx q[5],q[13];
x q[0];
cx q[15],q[9];
cx q[2],q[8];
h q[3];
s q[3];
h q[14];
x q[3];
h q[13];
cx q[13],q[12];
x q[11];
cx q[12],q[14];
h q[14];
s q[12];
cx q[7],q[13];
cx q[3],q[7];
cx q[15],q[1];
x q[14];
s q[2];
h q[2];
cx q[1],q[10];
t q[1];
s q[9];
cx q[6],q[8];
s q[0];
x q[13];
t q[10];
cx q[5],q[11];
cx q[4],q[2];
x q[9];
h q[3];
cx q[13],q[14];
s q[7];
s q[5];
x q[13];
h q[2];
h q[0];
x q[5];
t q[8];
s q[7];
x q[2];
cx q[14],q[10];
t q[0];
cx q[4],q[15];
t q[4];
cx q[1],q[13];
cx q[7],q[1];
cx q[13],q[10];
h q[8];
cx q[6],q[4];
cx q[11],q[1];
s q[14];
t q[3];
s q[9];
x q[10];
cx q[7],q[9];
cx q[13],q[8];
h q[12];
h q[10];
s q[8];
h q[5];
cx q[11],q[6];
t q[7];
t q[15];
t q[5];
cx q[9],q[3];
cx q[0],q[12];
x q[11];
cx q[8],q[13];
cx q[13],q[4];
s q[14];
cx q[7],q[11];
x q[6];
cx q[7],q[4];
h q[12];
cx q[0],q[5];
t q[5];
h q[13];
cx q[9],q[4];
s q[13];
x q[13];
t q[5];
t q[2];
cx q[12],q[5];
cx q[0],q[12];
s q[1];
cx q[2],q[13];
cx q[8],q[1];
cx q[1],q[7];
cx q[8],q[1];
t q[11];
t q[3];
cx q[6],q[14];
t q[8];
cx q[11],q[4];
cx q[6],q[4];
t q[9];
t q[3];
cx q[0],q[4];
h q[13];
cx q[12],q[0];
x q[2];
cx q[13],q[7];
cx q[2],q[13];
cx q[13],q[11];
cx q[11],q[0];
s q[0];
s q[4];
cx q[8],q[2];
cx q[6],q[0];
x q[12];
x q[10];
cx q[0],q[8];
cx q[15],q[7];
cx q[11],q[0];
cx q[0],q[3];
t q[11];
cx q[1],q[5];
cx q[14],q[0];
cx q[13],q[8];
h q[6];
h q[6];
x q[6];
cx q[6],q[11];
cx q[14],q[13];
t q[12];
cx q[0],q[7];
cx q[2],q[3];
cx q[5],q[12];
x q[14];
cx q[11],q[7];
x q[3];
cx q[9],q[0];
s q[1];
h q[10];
cx q[11],q[14];
s q[10];
cx q[1],q[6];
x q[1];
x q[6];
s q[8];
cx q[12],q[5];
t q[5];
cx q[7],q[13];
cx q[8],q[6];
s q[1];
t q[15];
s q[11];
cx q[5],q[10];
x q[7];
cx q[9],q[13];
cx q[5],q[6];
t q[15];
cx q[1],q[9];
cx q[10],q[14];
cx q[2],q[6];
h q[8];
t q[0];
x q[9];
cx q[3],q[2];
cx q[1],q[7];
s q[14];
t q[14];
cx q[7],q[9];
cx q[13],q[8];
cx q[13],q[2];
cx q[4],q[11];
cx q[15],q[3];
h q[11];
cx q[8],q[5];
t q[12];
s q[6];
cx q[7],q[4];
h q[12];
x q[7];
cx q[13],q[3];
cx q[10],q[11];
cx q[0],q[13];
s q[2];
cx q[3],q[14];
cx q[14],q[13];
cx q[5],q[2];
h q[13];
h q[11];
t q[14];
cx q[3],q[13];
cx q[11],q[1];
h q[15];
cx q[0],q[10];
cx q[1],q[3];
cx q[9],q[12];
x q[5];
cx q[5],q[0];
cx q[5],q[13];
cx q[9],q[15];
s q[4];
cx q[2],q[9];
s q[7];
t q[14];
x q[0];
x q[12];
cx q[0],q[6];
cx q[0],q[1];
cx q[1],q[5];
cx q[15],q[9];
cx q[15],q[11];
cx q[9],q[1];
h q[12];
t q[7];
cx q[7],q[2];
cx q[2],q[0];
cx q[14],q[13];
cx q[5],q[8];
x q[1];
x q[9];
cx q[13],q[1];
h q[9];
cx q[3],q[6];
x q[10];
cx q[15],q[9];
x q[6];
cx q[7],q[8];
s q[9];
cx q[15],q[2];
cx q[2],q[0];
cx q[9],q[10];
x q[14];
cx q[5],q[9];
cx q[2],q[10];
x q[9];
t q[7];
s q[3];
cx q[12],q[1];
h q[3];
cx q[4],q[10];
s q[4];
cx q[6],q[10];
cx q[6],q[14];
cx q[2],q[1];
cx q[13],q[14];
cx q[1],q[5];
s q[9];
cx q[6],q[2];
cx q[4],q[5];
s q[13];
cx q[15],q[4];
cx q[4],q[8];
cx q[3],q[1];
cx q[1],q[11];
cx q[6],q[10];
cx q[15],q[1];
cx q[6],q[5];
cx q[14],q[9];
cx q[10],q[3];
s q[3];
x q[10];
cx q[7],q[14];
s q[3];
t q[14];
s q[7];
cx q[8],q[4];
t q[11];
cx q[10],q[0];
cx q[2],q[11];
t q[3];
h q[14];
cx q[9],q[0];
cx q[11],q[13];
x q[14];
cx q[14],q[9];
cx q[9],q[8];
x q[10];
h q[1];
x q[1];
x q[14];
t q[12];
cx q[12],q[13];
t q[6];
t q[10];
cx q[2],q[13];
cx q[15],q[11];
cx q[14],q[7];
cx q[13],q[8];
cx q[5],q[15];
cx q[13],q[6];
h q[11];
cx q[11],q[10];
cx q[4],q[12];
x q[13];
x q[2];
s q[11];
s q[6];
s q[0];
cx q[12],q[5];
s q[14];
cx q[6],q[9];
cx q[5],q[1];
cx q[0],q[9];
h q[11];
t q[8];
s q[9];
x q[0];
cx q[1],q[8];
cx q[6],q[13];
t q[1];
s q[10];
h q[1];
s q[10];
cx q[15],q[14];
cx q[11],q[12];
t q[15];
h q[4];
h q[15];
cx q[0],q[14];
cx q[1],q[0];
cx q[13],q[3];
s q[2];
s q[15];
h q[4];
x q[12];
cx q[6],q[2];
x q[14];
h q[2];
s q[9];
cx q[12],q[3];
t q[13];
h q[0];
x q[11];
cx q[4],q[6];
h q[0];
cx q[8],q[7];
cx q[2],q[12];
x q[15];
cx q[12],q[0];
s q[13];
cx q[5],q[10];
s q[14];
cx q[4],q[15];
s q[13];
s q[4];
cx q[12],q[13];
t q[12];
t q[14];
t q[13];
x q[4];
s q[8];
t q[4];
cx q[8],q[13];
s q[13];
cx q[12],q[13];
t q[0];
cx q[3],q[13];
cx q[13],q[14];
t q[11];
x q[12];
cx q[11],q[6];
x q[14];
x q[8];
s q[13];
x q[8];
cx q[13],q[3];
h q[12];
cx q[3],q[9];
cx q[15],q[13];
cx q[7],q[5];
h q[9];
h q[12];
cx q[5],q[4];
cx q[0],q[13];
cx q[8],q[15];
cx q[3],q[1];
x q[9];
cx q[13],q[12];
cx q[12],q[9];
cx q[10],q[12];
x q[10];
s q[15];
cx q[4],q[5];
t q[14];
x q[12];
cx q[10],q[9];
cx q[15],q[6];
h q[14];
cx q[11],q[15];
s q[7];
cx q[15],q[14];
h q[2];
cx q[10],q[9];
cx q[10],q[9];
t q[15];
h q[1];
t q[14];
x q[8];
cx q[15],q[0];
t q[7];
t q[5];
h q[11];
x q[15];
s q[2];
cx q[11],q[13];
cx q[0],q[15];
x q[9];
cx q[5],q[3];
cx q[14],q[13];
s q[5];
cx q[9],q[2];
t q[13];
h q[10];
x q[9];
s q[0];
x q[14];
s q[1];
cx q[9],q[6];
h q[7];
h q[13];
t q[5];
s q[14];
cx q[4],q[6];
s q[3];
h q[3];
cx q[0],q[1];
x q[4];
cx q[11],q[5];
cx q[11],q[9];
cx q[12],q[3];
cx q[12],q[15];
cx q[2],q[6];
cx q[9],q[3];
cx q[10],q[15];
cx q[1],q[11];
h q[8];
cx q[13],q[3];
h q[2];
cx q[9],q[6];
x q[5];
s q[7];
cx q[1],q[0];
h q[3];
cx q[4],q[9];
s q[10];
cx q[8],q[7];
cx q[15],q[11];
s q[8];
cx q[3],q[7];
s q[10];
cx q[0],q[12];
cx q[7],q[3];
cx q[12],q[7];
cx q[2],q[13];
s q[4];
x q[14];
cx q[11],q[5];
cx q[4],q[3];
cx q[2],q[13];
h q[13];
h q[13];
h q[13];
cx q[12],q[6];
t q[5];
t q[7];
h q[10];
cx q[13],q[0];
cx q[6],q[8];
cx q[2],q[10];
cx q[2],q[6];
cx q[9],q[2];
cx q[11],q[14];
h q[14];
h q[1];
t q[6];
cx q[2],q[7];
x q[8];
h q[10];
cx q[15],q[9];
cx q[14],q[12];
s q[4];
x q[1];
cx q[14],q[1];
t q[11];
cx q[11],q[7];
cx q[4],q[7];
s q[12];
cx q[13],q[10];
cx q[15],q[1];
cx q[0],q[6];
cx q[0],q[14];
cx q[14],q[2];
cx q[6],q[12];
cx q[2],q[4];
cx q[7],q[14];
h q[2];
cx q[11],q[9];
cx q[14],q[7];
t q[11];
cx q[13],q[1];
x q[9];
h q[6];
cx q[4],q[10];
t q[8];
cx q[6],q[3];
s q[0];
cx q[3],q[9];
t q[1];
cx q[3],q[4];
x q[14];
cx q[6],q[5];
cx q[14],q[4];
cx q[3],q[1];
cx q[12],q[10];
cx q[12],q[0];
s q[15];
h q[2];
h q[0];
h q[7];
cx q[1],q[13];
cx q[14],q[3];
cx q[14],q[8];
h q[7];
t q[10];
t q[11];
h q[4];
t q[0];
cx q[8],q[10];
s q[2];
cx q[12],q[13];
x q[3];
s q[13];
x q[4];
cx q[7],q[11];
cx q[14],q[6];
cx q[12],q[10];
h q[14];
cx q[12],q[1];
cx q[1],q[14];
s q[12];
cx q[15],q[1];
cx q[13],q[9];
x q[11];
cx q[14],q[8];
t q[2];
cx q[12],q[3];
cx q[14],q[3];
s q[14];
cx q[11],q[12];
cx q[12],q[14];
h q[5];
x q[12];
x q[11];
x q[3];
x q[15];
t q[14];
h q[10];
cx q[13],q[6];
cx q[8],q[9];
cx q[11],q[5];
h q[4];
x q[8];
x q[6];
t q[3];
cx q[10],q[15];
h q[3];
t q[9];
cx q[11],q[8];
cx q[9],q[1];
x q[13];
cx q[13],q[2];
s q[5];
x q[9];
cx q[2],q[7];
x q[11];
t q[15];
cx q[2],q[7];
t q[3];
cx q[5],q[0];
cx q[2],q[13];
cx q[13],q[11];
cx q[5],q[11];
x q[7];
cx q[3],q[7];
t q[13];
h q[12];
cx q[5],q[4];
cx q[9],q[7];
cx q[3],q[9];
cx q[11],q[15];
cx q[3],q[6];
cx q[15],q[14];
cx q[3],q[4];
x q[7];
cx q[4],q[5];
x q[11];
cx q[7],q[14];
x q[5];
t q[11];
h q[12];
cx q[13],q[11];
cx q[8],q[15];
cx q[3],q[11];
cx q[8],q[3];
s q[9];
cx q[5],q[6];
x q[3];
cx q[4],q[3];
cx q[4],q[11];
h q[15];
cx q[2],q[11];
h q[14];
h q[4];
cx q[7],q[12];
cx q[12],q[8];
cx q[6],q[12];
cx q[15],q[14];
x q[7];
cx q[0],q[6];
t q[7];
s q[11];
cx q[6],q[11];
h q[8];
x q[15];
t q[2];
s q[13];
s q[13];
cx q[14],q[0];
cx q[7],q[0];
x q[15];
cx q[1],q[3];
x q[13];
t q[0];
cx q[1],q[3];
s q[12];
cx q[4],q[1];
cx q[8],q[14];
cx q[7],q[2];
s q[5];
cx q[4],q[3];
x q[1];
cx q[3],q[11];
h q[4];
h q[1];
cx q[6],q[7];
cx q[11],q[8];
x q[11];
t q[11];
h q[14];
t q[2];
cx q[1],q[13];
s q[15];
cx q[7],q[11];
h q[13];
cx q[7],q[15];
t q[3];
x q[9];
cx q[3],q[6];
s q[11];
cx q[7],q[14];
cx q[4],q[0];
x q[15];
cx q[12],q[1];
cx q[5],q[0];
cx q[7],q[4];
s q[3];
s q[14];
cx q[8],q[11];
t q[12];
cx q[12],q[7];
cx q[11],q[15];
cx q[3],q[4];
h q[1];
cx q[8],q[3];
x q[7];
h q[13];
s q[15];
h q[11];
h q[12];
t q[4];
cx q[0],q[13];
cx q[10],q[11];
h q[15];
cx q[15],q[3];
cx q[5],q[2];
cx q[14],q[13];
h q[5];
cx q[10],q[9];
s q[9];
cx q[4],q[8];
cx q[2],q[0];
cx q[12],q[2];
t q[4];
cx q[14],q[11];
cx q[3],q[14];
h q[14];
x q[4];
cx q[9],q[10];
cx q[12],q[6];
cx q[4],q[2];
t q[9];
cx q[8],q[3];
cx q[0],q[11];
s q[8];
cx q[9],q[12];
cx q[6],q[13];
x q[13];
h q[11];
cx q[13],q[1];
s q[6];
s q[5];
cx q[10],q[3];
t q[3];
h q[4];
x q[1];
s q[9];
cx q[0],q[6];
cx q[15],q[0];
cx q[11],q[5];
cx q[5],q[15];
t q[9];
t q[14];
cx q[0],q[14];
cx q[10],q[4];
cx q[14],q[3];
cx q[3],q[5];
cx q[15],q[2];
x q[0];
t q[4];
x q[14];
h q[15];
cx q[4],q[6];
x q[14];
cx q[15],q[2];
cx q[13],q[15];
t q[7];
t q[13];
cx q[9],q[1];
cx q[9],q[11];
cx q[5],q[7];
t q[9];
s q[6];
s q[8];
cx q[0],q[5];
x q[4];
x q[3];
cx q[8],q[1];
cx q[4],q[6];
x q[5];
x q[0];
s q[14];
t q[12];
cx q[7],q[15];
cx q[6],q[11];
cx q[14],q[5];
h q[6];
cx q[0],q[4];
cx q[1],q[11];
h q[15];
t q[11];
cx q[10],q[9];
h q[9];
x q[5];
s q[10];
cx q[8],q[1];
cx q[14],q[3];
cx q[7],q[12];
x q[7];
s q[10];
s q[8];
cx q[7],q[6];
cx q[8],q[13];
s q[7];
cx q[9],q[13];
cx q[8],q[3];
cx q[12],q[11];
cx q[2],q[15];
cx q[11],q[4];
t q[2];
x q[15];
cx q[10],q[7];
s q[6];
cx q[1],q[0];
s q[11];
cx q[3],q[12];
cx q[10],q[11];
cx q[14],q[8];
t q[4];
s q[15];
h q[8];
cx q[7],q[11];
s q[2];
x q[7];
cx q[8],q[14];
s q[8];
x q[0];
cx q[15],q[13];
cx q[7],q[14];
s q[6];
cx q[14],q[15];
cx q[8],q[11];
x q[9];
x q[4];
cx q[12],q[1];
x q[15];
cx q[11],q[3];
h q[10];
cx q[6],q[9];
t q[1];
t q[11];
t q[13];
x q[14];
s q[9];
t q[11];
h q[4];
cx q[6],q[13];
s q[7];
h q[6];
cx q[15],q[0];
cx q[13],q[7];
cx q[8],q[5];
x q[8];
cx q[12],q[9];
cx q[0],q[13];
cx q[11],q[10];
t q[14];
cx q[11],q[13];
s q[7];
cx q[1],q[14];
cx q[13],q[8];
cx q[2],q[6];
cx q[11],q[15];
cx q[11],q[10];
cx q[3],q[9];
s q[7];
x q[2];
t q[11];
cx q[10],q[3];
h q[9];
cx q[7],q[0];
cx q[11],q[8];
cx q[1],q[12];
cx q[14],q[5];
t q[15];
cx q[9],q[15];
x q[7];
cx q[14],q[10];
h q[9];
x q[8];
s q[11];
cx q[0],q[4];
s q[5];
cx q[0],q[4];
x q[5];
x q[10];
cx q[14],q[11];
cx q[2],q[3];
cx q[6],q[11];
cx q[13],q[11];
cx q[4],q[11];
cx q[11],q[0];
cx q[6],q[2];
s q[11];
x q[14];
cx q[6],q[10];
cx q[0],q[2];
cx q[11],q[3];
cx q[3],q[10];
t q[9];
h q[11];
cx q[1],q[5];
cx q[15],q[3];
t q[3];
s q[7];
s q[2];
h q[4];
s q[10];
x q[15];
cx q[14],q[8];
cx q[6],q[9];
cx q[3],q[14];
cx q[6],q[10];
cx q[7],q[4];
h q[0];
x q[0];
x q[10];
cx q[1],q[14];
s q[2];
cx q[10],q[11];
x q[9];
cx q[14],q[6];
cx q[10],q[13];
cx q[6],q[2];
cx q[10],q[1];
t q[9];
cx q[15],q[8];
cx q[5],q[15];
t q[2];
h q[12];
t q[3];
h q[4];
t q[12];
cx q[8],q[0];
cx q[11],q[4];
cx q[0],q[1];
cx q[9],q[7];
cx q[2],q[8];
t q[3];
x q[0];
x q[5];
cx q[15],q[2];
h q[2];
s q[4];
h q[12];
s q[0];
t q[15];
cx q[8],q[10];
s q[15];
h q[1];
x q[14];
h q[3];
x q[3];
s q[8];
s q[8];
s q[5];
cx q[11],q[12];
cx q[3],q[12];
cx q[14],q[4];
t q[11];
cx q[11],q[5];
s q[11];
s q[8];
cx q[10],q[11];
t q[9];
t q[12];
x q[1];
cx q[2],q[3];
t q[11];
cx q[14],q[10];
cx q[1],q[9];
cx q[2],q[1];
cx q[5],q[6];
t q[11];
cx q[10],q[13];
cx q[13],q[4];
h q[5];
cx q[11],q[5];
cx q[15],q[2];
cx q[8],q[15];
cx q[0],q[9];
t q[6];
cx q[6],q[4];
cx q[7],q[8];
x q[1];
t q[2];
cx q[10],q[7];
cx q[11],q[7];
x q[7];
cx q[9],q[0];
s q[8];
cx q[15],q[11];
cx q[5],q[8];
h q[0];
cx q[13],q[10];
cx q[13],q[12];
cx q[13],q[10];
s q[6];
x q[3];
s q[12];
s q[12];
cx q[14],q[15];
cx q[15],q[6];
h q[14];
cx q[2],q[1];
cx q[11],q[8];
cx q[11],q[4];
cx q[14],q[9];
x q[14];
cx q[5],q[7];
cx q[15],q[14];
cx q[13],q[14];
cx q[4],q[12];
cx q[14],q[5];
t q[1];
x q[7];
h q[11];
cx q[0],q[6];
t q[0];
cx q[3],q[11];
h q[1];
cx q[0],q[12];
x q[2];
cx q[14],q[8];
cx q[5],q[6];
cx q[2],q[10];
cx q[11],q[4];
cx q[8],q[3];
s q[14];
s q[12];
h q[10];
cx q[7],q[3];
x q[1];
x q[11];
t q[5];
h q[3];
h q[12];
x q[7];
cx q[4],q[10];
cx q[12],q[4];
t q[4];
s q[10];
cx q[0],q[1];
cx q[4],q[2];
s q[5];
cx q[11],q[12];
cx q[10],q[3];
cx q[11],q[14];
x q[3];
h q[3];
h q[5];
cx q[3],q[2];
cx q[14],q[6];
cx q[5],q[8];
cx q[13],q[3];
cx q[3],q[4];
s q[7];
x q[13];
cx q[2],q[1];
x q[6];
s q[3];
x q[8];
h q[4];
cx q[12],q[4];
s q[14];
cx q[9],q[15];
cx q[2],q[7];
s q[4];
h q[13];
cx q[13],q[12];
s q[0];
t q[0];
cx q[6],q[11];
h q[6];
h q[11];
cx q[1],q[11];
x q[4];
cx q[1],q[9];
x q[0];
s q[14];
h q[3];
cx q[6],q[11];
cx q[2],q[3];
h q[9];
t q[8];
x q[2];
cx q[8],q[7];
h q[14];
s q[10];
cx q[9],q[13];
h q[5];
cx q[10],q[3];
x q[1];
t q[9];
cx q[1],q[3];
cx q[2],q[11];
cx q[8],q[4];
cx q[14],q[0];
h q[12];
cx q[6],q[11];
x q[2];
cx q[2],q[8];
t q[12];
s q[8];
cx q[12],q[6];
t q[14];
h q[14];
h q[14];
h q[15];
h q[5];
cx q[12],q[2];
x q[15];
cx q[7],q[12];
cx q[7],q[2];
t q[0];
cx q[2],q[7];
s q[4];
x q[8];
h q[0];
t q[14];
x q[1];
t q[13];
h q[12];
h q[14];
x q[2];
s q[3];
x q[4];
h q[15];
h q[6];
t q[3];
t q[8];
t q[15];
