// Initial wiring: [12, 4, 2, 14, 0, 9, 1, 15, 6, 8, 10, 11, 5, 13, 3, 7]
// Resulting wiring: [12, 4, 2, 14, 0, 9, 1, 15, 6, 8, 10, 11, 5, 13, 3, 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[1], q[0];
cx q[5], q[2];
cx q[6], q[5];
cx q[8], q[7];
cx q[11], q[10];
cx q[10], q[5];
cx q[5], q[2];
cx q[13], q[12];
cx q[13], q[10];
cx q[12], q[11];
cx q[10], q[5];
cx q[15], q[14];
cx q[14], q[15];
cx q[10], q[13];
cx q[9], q[10];
cx q[10], q[13];
cx q[13], q[10];
cx q[0], q[7];
