// Initial wiring: [12, 1, 6, 4, 2, 5, 9, 15, 3, 14, 0, 7, 13, 10, 11, 8]
// Resulting wiring: [12, 1, 6, 4, 2, 5, 9, 15, 3, 14, 0, 7, 13, 10, 11, 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[3], q[2];
cx q[6], q[5];
cx q[11], q[4];
cx q[4], q[3];
cx q[3], q[2];
cx q[2], q[1];
cx q[4], q[3];
cx q[12], q[11];
cx q[11], q[4];
cx q[13], q[10];
cx q[14], q[15];
cx q[11], q[12];
cx q[4], q[11];
cx q[2], q[3];
cx q[3], q[4];
cx q[1], q[2];
cx q[2], q[3];
cx q[3], q[4];
cx q[4], q[11];
cx q[11], q[12];
cx q[11], q[4];
cx q[12], q[11];
