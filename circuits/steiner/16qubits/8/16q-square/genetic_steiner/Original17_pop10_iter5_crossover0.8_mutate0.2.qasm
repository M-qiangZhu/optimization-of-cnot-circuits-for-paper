// Initial wiring: [6, 9, 5, 13, 0, 12, 1, 10, 15, 4, 3, 7, 8, 14, 11, 2]
// Resulting wiring: [6, 9, 5, 13, 0, 12, 1, 10, 15, 4, 3, 7, 8, 14, 11, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[9], q[6];
cx q[9], q[8];
cx q[6], q[1];
cx q[11], q[10];
cx q[10], q[9];
cx q[10], q[5];
cx q[11], q[10];
cx q[12], q[11];
cx q[11], q[10];
cx q[10], q[9];
cx q[9], q[8];
cx q[10], q[5];
cx q[11], q[10];
cx q[12], q[11];
cx q[14], q[13];
cx q[13], q[10];
cx q[10], q[5];
cx q[13], q[14];
cx q[10], q[13];
cx q[5], q[10];
cx q[4], q[5];
cx q[5], q[10];
cx q[10], q[13];
cx q[13], q[14];
cx q[10], q[5];
cx q[13], q[10];
cx q[1], q[6];
cx q[0], q[1];
cx q[1], q[6];
