OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[2],q[1];
cx q[19],q[18];
cx q[11],q[10];
cx q[10],q[5];
cx q[11],q[10];
