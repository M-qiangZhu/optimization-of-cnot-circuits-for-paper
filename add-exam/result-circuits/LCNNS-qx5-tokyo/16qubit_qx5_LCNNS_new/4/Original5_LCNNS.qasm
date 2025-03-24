OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[9],q[10];
cx q[2],q[13];
cx q[11],q[4];
cx q[13],q[12];
