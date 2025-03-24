OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[1],q[0];
cx q[6],q[1];
cx q[1],q[0];
cx q[13],q[14];
cx q[9],q[4];
