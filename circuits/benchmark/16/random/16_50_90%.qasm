OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2],q[1];
cx q[9],q[6];
x q[5];
cx q[14],q[0];
cx q[14],q[4];
cx q[0],q[10];
cx q[8],q[6];
cx q[10],q[14];
cx q[11],q[15];
cx q[3],q[8];
t q[14];
cx q[4],q[6];
cx q[5],q[4];
cx q[12],q[13];
cx q[15],q[8];
h q[15];
cx q[10],q[9];
cx q[8],q[11];
cx q[6],q[0];
cx q[14],q[4];
cx q[3],q[9];
s q[4];
cx q[2],q[11];
cx q[1],q[9];
cx q[2],q[3];
cx q[9],q[0];
cx q[1],q[5];
cx q[1],q[12];
s q[13];
cx q[2],q[12];
cx q[7],q[9];
cx q[8],q[0];
cx q[1],q[10];
cx q[5],q[1];
cx q[3],q[9];
cx q[9],q[11];
cx q[7],q[0];
cx q[4],q[1];
cx q[11],q[0];
cx q[1],q[7];
s q[0];
cx q[2],q[14];
cx q[5],q[9];
cx q[0],q[13];
cx q[4],q[10];
cx q[14],q[0];
cx q[12],q[14];
cx q[3],q[15];
cx q[8],q[4];
cx q[15],q[2];
