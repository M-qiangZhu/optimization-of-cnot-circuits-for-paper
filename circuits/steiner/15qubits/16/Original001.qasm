// Initial wiring: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
// Resulting wiring: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[15];
cx q[11], q[4];
cx q[6], q[9];
cx q[12], q[10];
cx q[13], q[9];
cx q[1], q[10];
cx q[12], q[14];
cx q[7], q[10];
cx q[11], q[14];
cx q[0], q[10];
cx q[13], q[3];
cx q[0], q[10];
cx q[12], q[9];
cx q[2], q[13];
cx q[7], q[8];
cx q[2], q[12];
cx q[14], q[1];
