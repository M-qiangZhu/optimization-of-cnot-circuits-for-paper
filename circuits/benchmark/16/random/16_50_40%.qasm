OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
x q[14];
t q[11];
s q[10];
cx q[12],q[15];
s q[10];
s q[2];
cx q[14],q[10];
s q[11];
t q[12];
cx q[8],q[1];
h q[2];
t q[12];
s q[9];
t q[7];
cx q[11],q[0];
x q[6];
x q[2];
h q[5];
x q[12];
cx q[13],q[0];
x q[1];
x q[5];
s q[5];
cx q[0],q[7];
s q[10];
t q[9];
h q[0];
h q[3];
x q[15];
h q[2];
cx q[6],q[5];
s q[14];
t q[8];
t q[4];
cx q[12],q[2];
cx q[10],q[4];
s q[14];
cx q[9],q[2];
cx q[4],q[15];
x q[0];
cx q[5],q[8];
h q[0];
x q[14];
cx q[2],q[5];
x q[0];
cx q[4],q[5];
cx q[12],q[11];
cx q[9],q[3];
s q[15];
cx q[7],q[12];
