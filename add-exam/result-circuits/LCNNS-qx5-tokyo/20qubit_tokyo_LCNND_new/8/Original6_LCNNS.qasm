OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
cx q[14],q[13];
cx q[19],q[14];
cx q[18],q[19];
cx q[14],q[13];
cx q[18],q[14];
cx q[12],q[13];
cx q[7],q[12];
cx q[6],q[7];
cx q[5],q[6];
cx q[6],q[7];
cx q[7],q[12];
cx q[12],q[13];
cx q[12],q[7];
cx q[7],q[12];
cx q[6],q[7];
cx q[5],q[6];
cx q[6],q[7];
cx q[1],q[6];
cx q[8],q[7];
cx q[7],q[8];
cx q[12],q[7];
cx q[11],q[12];
cx q[12],q[7];
cx q[7],q[8];
cx q[7],q[12];
cx q[11],q[12];
cx q[12],q[7];
