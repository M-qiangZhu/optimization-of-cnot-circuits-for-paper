OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
x q[5];
h q[10];
cx q[6],q[1];
h q[2];
x q[1];
x q[14];
cx q[13],q[8];
x q[9];
h q[14];
h q[0];
cx q[10],q[7];
s q[3];
h q[4];
x q[1];
t q[12];
t q[4];
cx q[5],q[9];
t q[9];
cx q[14],q[9];
t q[13];
