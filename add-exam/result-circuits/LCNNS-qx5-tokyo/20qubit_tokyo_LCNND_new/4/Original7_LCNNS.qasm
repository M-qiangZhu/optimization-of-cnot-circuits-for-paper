OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[5],q[10];
cx q[10],q[15];
cx q[0],q[1];
cx q[9],q[4];
