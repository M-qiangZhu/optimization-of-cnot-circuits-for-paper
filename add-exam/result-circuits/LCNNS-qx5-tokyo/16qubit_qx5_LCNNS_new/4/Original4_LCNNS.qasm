OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[9],q[6];
cx q[13],q[2];
cx q[11],q[12];
cx q[15],q[14];
