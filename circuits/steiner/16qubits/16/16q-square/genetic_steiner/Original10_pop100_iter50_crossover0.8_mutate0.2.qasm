// Initial wiring: [8, 1, 7, 3, 14, 5, 9, 13, 10, 2, 11, 0, 6, 15, 12, 4]
// Resulting wiring: [8, 1, 7, 3, 14, 5, 9, 13, 10, 2, 11, 0, 6, 15, 12, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[6], q[5];
cx q[7], q[6];
cx q[6], q[5];
cx q[9], q[8];
cx q[10], q[9];
cx q[9], q[8];
cx q[10], q[9];
cx q[15], q[14];
cx q[14], q[15];
cx q[12], q[13];
cx q[9], q[14];
cx q[14], q[15];
cx q[14], q[13];
cx q[8], q[9];
cx q[6], q[9];
cx q[5], q[10];
cx q[2], q[3];
cx q[3], q[2];
cx q[1], q[6];
cx q[6], q[9];
cx q[0], q[7];
