// Initial wiring: [1, 5, 7, 2, 0, 8, 6, 3, 4]
// Resulting wiring: [1, 5, 7, 2, 0, 8, 6, 3, 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[1], q[2];
cx q[0], q[1];
cx q[1], q[2];
cx q[4], q[5];
cx q[6], q[7];
cx q[7], q[8];
cx q[4], q[3];
