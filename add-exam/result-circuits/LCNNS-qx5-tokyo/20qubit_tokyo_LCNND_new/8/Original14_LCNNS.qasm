OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[18],q[19];
cx q[15],q[16];
cx q[10],q[11];
cx q[5],q[0];
cx q[6],q[5];
cx q[5],q[0];
cx q[7],q[6];
cx q[6],q[5];
cx q[16],q[11];
cx q[11],q[16];
cx q[12],q[11];
cx q[11],q[16];
cx q[7],q[12];
cx q[12],q[11];
cx q[7],q[6];
cx q[7],q[12];
