OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[7],q[12];
cx q[11],q[5];
cx q[5],q[11];
cx q[10],q[5];
cx q[8],q[9];
cx q[9],q[8];
cx q[3],q[9];
cx q[4],q[3];
cx q[17],q[16];
cx q[5],q[0];
cx q[5],q[11];
cx q[0],q[5];
cx q[1],q[0];
cx q[9],q[8];
cx q[8],q[9];
cx q[0],q[5];
cx q[5],q[0];
cx q[11],q[5];
