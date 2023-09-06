// Initial wiring: [5, 0, 10, 1, 14, 7, 12, 9, 3, 6, 8, 4, 11, 15, 2, 13]
// Resulting wiring: [5, 0, 10, 1, 14, 7, 12, 9, 3, 6, 8, 4, 11, 15, 2, 13]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[3], q[2];
cx q[5], q[4];
cx q[6], q[5];
cx q[5], q[4];
cx q[5], q[2];
cx q[6], q[5];
cx q[9], q[6];
cx q[11], q[4];
cx q[12], q[11];
cx q[11], q[4];
cx q[4], q[3];
cx q[14], q[9];
cx q[9], q[8];
cx q[8], q[7];
cx q[9], q[6];
cx q[9], q[8];
cx q[14], q[9];
cx q[15], q[8];
cx q[8], q[7];
cx q[14], q[15];
cx q[6], q[7];
cx q[5], q[10];
