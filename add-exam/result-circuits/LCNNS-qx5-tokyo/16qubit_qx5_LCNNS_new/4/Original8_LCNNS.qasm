OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[8],q[7];
cx q[13],q[2];
cx q[14],q[1];
cx q[13],q[14];
