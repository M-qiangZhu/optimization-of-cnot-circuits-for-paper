OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[1],q[6];
cx q[6],q[1];
cx q[10],q[6];
cx q[15],q[10];
cx q[6],q[1];
cx q[2],q[1];
cx q[3],q[2];
cx q[9],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[1],q[6];
cx q[2],q[1];
cx q[3],q[2];
cx q[9],q[3];
cx q[3],q[2];
cx q[2],q[1];
cx q[14],q[18];
cx q[2],q[3];
cx q[3],q[2];
cx q[9],q[3];
cx q[3],q[2];
cx q[2],q[3];
cx q[8],q[7];
cx q[5],q[11];
cx q[12],q[13];
