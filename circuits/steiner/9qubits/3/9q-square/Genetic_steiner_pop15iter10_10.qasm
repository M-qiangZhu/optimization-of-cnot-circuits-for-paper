// Initial wiring: [4, 8, 7, 5, 1, 2, 3, 6, 0]
// Resulting wiring: [4, 8, 7, 5, 1, 2, 3, 6, 0]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[1];
cx q[7], q[4];
cx q[3], q[4];
