// Initial wiring: [0 4 2 8 3 5 6 7 1]
// Resulting wiring: [0 4 1 8 3 5 7 6 2]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
cx q[0], q[1];
cx q[1], q[2];
cx q[1], q[2];
cx q[1], q[2];
cx q[4], q[7];
cx q[6], q[7];
cx q[6], q[7];
cx q[6], q[7];
cx q[7], q[8];
cx q[0], q[1];
cx q[8], q[3];
