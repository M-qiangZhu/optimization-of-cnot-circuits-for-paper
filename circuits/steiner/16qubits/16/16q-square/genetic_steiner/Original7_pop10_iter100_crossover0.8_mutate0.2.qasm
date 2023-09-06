// Initial wiring: [14, 6, 11, 8, 7, 3, 10, 2, 13, 1, 5, 15, 0, 9, 4, 12]
// Resulting wiring: [14, 6, 11, 8, 7, 3, 10, 2, 13, 1, 5, 15, 0, 9, 4, 12]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[2], q[1];
cx q[9], q[8];
cx q[10], q[5];
cx q[10], q[9];
cx q[5], q[2];
cx q[9], q[8];
cx q[2], q[1];
cx q[10], q[9];
cx q[10], q[5];
cx q[13], q[10];
cx q[10], q[5];
cx q[5], q[2];
cx q[10], q[5];
cx q[13], q[10];
cx q[14], q[15];
cx q[10], q[11];
cx q[9], q[10];
cx q[6], q[9];
cx q[9], q[10];
cx q[10], q[11];
cx q[9], q[8];
cx q[5], q[10];
cx q[4], q[11];
cx q[4], q[5];
cx q[3], q[4];
cx q[2], q[3];
