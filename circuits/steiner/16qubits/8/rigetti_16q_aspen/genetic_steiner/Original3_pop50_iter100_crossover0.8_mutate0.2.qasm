// Initial wiring: [0, 13, 1, 6, 7, 14, 3, 4, 12, 15, 5, 2, 8, 11, 10, 9]
// Resulting wiring: [0, 13, 1, 6, 7, 14, 3, 4, 12, 15, 5, 2, 8, 11, 10, 9]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
cx q[7], q[0];
cx q[11], q[10];
cx q[14], q[13];
cx q[13], q[12];
cx q[12], q[11];
cx q[13], q[12];
cx q[8], q[9];
cx q[6], q[7];
cx q[0], q[1];
