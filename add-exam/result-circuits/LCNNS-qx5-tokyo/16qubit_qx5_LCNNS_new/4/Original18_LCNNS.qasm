OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[4],q[5];
cx q[1],q[14];
cx q[11],q[10];
cx q[10],q[5];
