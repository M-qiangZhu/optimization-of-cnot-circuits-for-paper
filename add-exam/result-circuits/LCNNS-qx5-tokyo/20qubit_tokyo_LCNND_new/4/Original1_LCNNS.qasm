OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[19],q[18];
cx q[2],q[3];
cx q[1],q[2];
cx q[2],q[3];
cx q[1],q[2];
cx q[15],q[10];
cx q[5],q[10];
