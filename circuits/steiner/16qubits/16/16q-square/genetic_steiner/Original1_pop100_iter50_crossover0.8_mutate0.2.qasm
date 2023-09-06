// Initial wiring: [6, 13, 0, 5, 7, 3, 2, 8, 12, 14, 4, 10, 11, 1, 9, 15]
// Resulting wiring: [6, 13, 0, 5, 7, 3, 2, 8, 12, 14, 4, 10, 11, 1, 9, 15]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[3], q[2];
cx q[6], q[5];
cx q[7], q[0];
cx q[9], q[8];
cx q[10], q[9];
cx q[9], q[8];
cx q[8], q[7];
cx q[10], q[5];
cx q[12], q[11];
cx q[14], q[9];
cx q[15], q[8];
cx q[8], q[7];
cx q[7], q[6];
cx q[6], q[5];
cx q[5], q[4];
cx q[7], q[0];
cx q[15], q[8];
cx q[12], q[13];
cx q[11], q[12];
cx q[12], q[13];
cx q[13], q[12];
cx q[8], q[15];
