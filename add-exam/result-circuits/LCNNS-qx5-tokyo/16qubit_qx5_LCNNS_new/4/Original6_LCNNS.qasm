OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[9],q[6];
cx q[12],q[11];
cx q[10],q[5];
cx q[11],q[10];
