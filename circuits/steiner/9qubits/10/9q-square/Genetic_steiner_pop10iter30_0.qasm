// Initial wiring: [6, 0, 2, 4, 3, 7, 8, 5, 1]
// Resulting wiring: [6, 0, 2, 4, 3, 7, 8, 5, 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[5], q[6];
cx q[6], q[7];
cx q[5], q[6];
cx q[7], q[8];
cx q[5], q[4];
cx q[6], q[5];
cx q[5], q[6];
cx q[8], q[3];
cx q[3], q[2];
cx q[2], q[1];
cx q[5], q[0];
