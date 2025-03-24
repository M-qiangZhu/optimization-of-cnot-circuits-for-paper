OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[0],q[5];
cx q[5],q[6];
cx q[17],q[16];
cx q[0],q[1];
