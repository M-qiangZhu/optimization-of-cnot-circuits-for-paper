// Initial wiring: [5 2 1 3 7 6 0 4 8]
// Resulting wiring: [5 2 1 3 7 6 0 4 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[4], q[5];
cx q[1], q[0];
cx q[2], q[1];
cx q[5], q[6];
cx q[4], q[3];
