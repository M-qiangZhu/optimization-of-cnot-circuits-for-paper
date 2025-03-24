OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2],q[3];
cx q[12],q[3];
cx q[4],q[3];
cx q[10],q[11];
