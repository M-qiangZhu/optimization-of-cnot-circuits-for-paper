OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[6],q[7];
cx q[9],q[6];
cx q[2],q[13];
cx q[1],q[14];
