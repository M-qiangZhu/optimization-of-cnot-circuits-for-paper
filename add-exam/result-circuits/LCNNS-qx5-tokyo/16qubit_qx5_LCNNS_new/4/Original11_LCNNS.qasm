OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[15],q[14];
cx q[5],q[10];
cx q[4],q[11];
cx q[2],q[3];
