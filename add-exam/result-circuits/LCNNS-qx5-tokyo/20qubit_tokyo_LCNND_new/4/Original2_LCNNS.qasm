OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[6],q[5];
cx q[7],q[6];
cx q[6],q[5];
cx q[1],q[2];
cx q[16],q[11];
