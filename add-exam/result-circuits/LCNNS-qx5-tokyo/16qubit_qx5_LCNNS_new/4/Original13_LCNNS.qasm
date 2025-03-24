OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2],q[1];
cx q[5],q[4];
cx q[11],q[4];
cx q[9],q[10];
