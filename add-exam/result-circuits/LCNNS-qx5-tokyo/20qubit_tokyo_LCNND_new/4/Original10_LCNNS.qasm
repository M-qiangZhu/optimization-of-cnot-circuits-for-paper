OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[10],q[15];
cx q[5],q[10];
cx q[19],q[18];
cx q[14],q[19];
cx q[14],q[18];
