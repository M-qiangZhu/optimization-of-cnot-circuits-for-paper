OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[9],q[8];
cx q[6],q[9];
cx q[5],q[10];
cx q[4],q[3];
