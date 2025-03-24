OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[8],q[9];
cx q[9],q[6];
cx q[8],q[7];
cx q[11],q[10];
