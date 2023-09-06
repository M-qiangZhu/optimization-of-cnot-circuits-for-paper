// Initial wiring: [5, 0, 15, 12, 2, 9, 4, 1, 7, 6, 8, 13, 10, 14, 3, 11]
// Resulting wiring: [5, 0, 15, 12, 2, 9, 4, 1, 7, 6, 8, 13, 10, 14, 3, 11]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[3], q[2];
cx q[2], q[1];
cx q[5], q[2];
cx q[2], q[1];
cx q[1], q[0];
cx q[2], q[1];
cx q[5], q[2];
cx q[6], q[5];
cx q[5], q[4];
cx q[6], q[5];
cx q[7], q[6];
cx q[6], q[5];
cx q[5], q[4];
cx q[6], q[1];
cx q[8], q[7];
cx q[9], q[6];
cx q[6], q[5];
cx q[5], q[4];
cx q[6], q[5];
cx q[9], q[6];
cx q[10], q[5];
cx q[5], q[2];
cx q[2], q[1];
cx q[1], q[0];
cx q[2], q[1];
cx q[5], q[2];
cx q[11], q[10];
cx q[10], q[5];
cx q[11], q[10];
cx q[12], q[11];
cx q[11], q[4];
cx q[12], q[11];
cx q[15], q[8];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[5];
cx q[5], q[4];
cx q[6], q[1];
cx q[7], q[0];
cx q[13], q[14];
cx q[1], q[6];
