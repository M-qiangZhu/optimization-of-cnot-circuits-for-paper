// Initial wiring: [0 1 2 3 4 6 7 5 8]
// Resulting wiring: [0 4 2 3 1 6 7 5 8]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[3], q[4];
cx q[1], q[4];
cx q[1], q[4];
cx q[1], q[4];
cx q[7], q[4];
cx q[5], q[4];
