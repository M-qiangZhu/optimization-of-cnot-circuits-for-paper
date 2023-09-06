// Initial wiring: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
// Resulting wiring: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[12], q[4];
cx q[9], q[7];
cx q[9], q[11];
cx q[9], q[10];
cx q[9], q[12];
cx q[11], q[6];
cx q[0], q[11];
cx q[10], q[6];
cx q[11], q[5];
cx q[13], q[0];
cx q[4], q[2];
cx q[7], q[9];
cx q[2], q[4];
cx q[11], q[1];
cx q[12], q[1];
cx q[14], q[7];
cx q[13], q[12];
cx q[6], q[15];
cx q[12], q[1];
cx q[3], q[11];
cx q[7], q[1];
cx q[13], q[5];
cx q[0], q[6];
cx q[13], q[4];
cx q[9], q[2];
cx q[14], q[6];
cx q[14], q[2];
cx q[4], q[8];
cx q[7], q[2];
cx q[12], q[3];
cx q[2], q[1];
cx q[11], q[2];
