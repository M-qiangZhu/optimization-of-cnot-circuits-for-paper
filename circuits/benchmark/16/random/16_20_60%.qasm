OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[10],q[6];
cx q[2],q[0];
s q[0];
cx q[15],q[7];
cx q[2],q[4];
cx q[14],q[1];
cx q[1],q[15];
h q[6];
h q[7];
t q[13];
t q[7];
t q[1];
t q[9];
s q[4];
x q[13];
s q[4];
cx q[0],q[5];
cx q[7],q[6];
h q[15];
cx q[14],q[4];
