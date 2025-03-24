OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[1],q[2];
cx q[12],q[3];
cx q[5],q[10];
cx q[13],q[12];
