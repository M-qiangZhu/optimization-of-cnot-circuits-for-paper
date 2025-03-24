OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[14],q[1];
cx q[11],q[4];
cx q[11],q[12];
cx q[6],q[5];
