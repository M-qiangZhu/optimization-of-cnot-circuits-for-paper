// Initial wiring: [15, 1, 13, 6, 5, 10, 4, 0, 14, 12, 3, 11, 9, 7, 2, 8]
// Resulting wiring: [15, 1, 13, 6, 5, 10, 4, 0, 14, 12, 3, 11, 9, 7, 2, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[6], q[5];
cx q[10], q[9];
cx q[11], q[10];
cx q[12], q[11];
cx q[11], q[10];
cx q[12], q[11];
cx q[14], q[13];
cx q[15], q[0];
cx q[9], q[10];
cx q[10], q[11];
cx q[7], q[8];
cx q[6], q[9];
cx q[9], q[10];
cx q[4], q[11];
cx q[3], q[4];
cx q[4], q[5];
cx q[0], q[1];
