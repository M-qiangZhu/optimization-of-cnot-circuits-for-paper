// Initial wiring: [5, 0, 7, 3, 6, 1, 4, 8, 2]
// Resulting wiring: [5, 0, 7, 3, 6, 1, 4, 8, 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[5];
cx q[1], q[4];
cx q[5], q[6];
cx q[4], q[7];
cx q[1], q[4];
cx q[4], q[7];
cx q[7], q[8];
cx q[6], q[5];
cx q[5], q[4];
cx q[5], q[0];
cx q[6], q[5];
cx q[5], q[6];
