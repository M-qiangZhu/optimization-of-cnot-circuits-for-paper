// Initial wiring: [14, 1, 9, 13, 5, 4, 2, 3, 7, 10, 12, 11, 8, 15, 6, 0]
// Resulting wiring: [14, 1, 9, 13, 5, 4, 2, 3, 7, 10, 12, 11, 8, 15, 6, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[3], q[2];
cx q[4], q[3];
cx q[6], q[5];
cx q[8], q[7];
cx q[10], q[9];
cx q[11], q[10];
cx q[10], q[9];
cx q[9], q[6];
cx q[11], q[4];
cx q[11], q[10];
cx q[12], q[11];
cx q[11], q[4];
cx q[4], q[3];
cx q[12], q[11];
cx q[14], q[13];
cx q[7], q[8];
cx q[5], q[10];
cx q[10], q[13];
cx q[3], q[4];
cx q[4], q[11];
cx q[1], q[6];
cx q[1], q[2];
