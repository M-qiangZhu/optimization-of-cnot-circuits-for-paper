OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2],q[13];
cx q[7],q[8];
cx q[10],q[9];
cx q[12],q[13];
