OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[15],q[14];
cx q[5],q[4];
cx q[9],q[6];
cx q[12],q[11];
