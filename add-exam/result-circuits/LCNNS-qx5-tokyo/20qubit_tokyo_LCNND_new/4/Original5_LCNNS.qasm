OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[2],q[3];
cx q[15],q[10];
cx q[11],q[16];
cx q[16],q[15];
cx q[11],q[16];
cx q[11],q[10];
