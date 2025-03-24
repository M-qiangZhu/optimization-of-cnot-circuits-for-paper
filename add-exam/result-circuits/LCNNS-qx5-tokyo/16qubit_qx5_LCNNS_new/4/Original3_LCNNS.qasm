OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[8],q[9];
cx q[6],q[7];
cx q[15],q[0];
cx q[3],q[12];
