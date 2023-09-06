// Initial wiring: [2, 3, 8, 0, 1, 4, 5, 6, 7]
// Resulting wiring: [2, 3, 8, 0, 1, 4, 5, 6, 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[2], q[3];
cx q[3], q[4];
cx q[7], q[8];
cx q[5], q[4];
cx q[1], q[0];
