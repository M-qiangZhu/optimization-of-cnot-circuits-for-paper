OPENQASM 2.0;
include "qelib1.inc";
qreg q[5];
creg meas[5];
cx q[3],q[1];
cx q[4],q[3];
cx q[3],q[1];
cx q[2],q[1];
cx q[1],q[0];
cx q[3],q[4];
cx q[1],q[3];
cx q[2],q[1];
cx q[4],q[3];
cx q[3],q[1];
cx q[1],q[2];
cx q[3],q[1];
cx q[3],q[4];
cx q[1],q[3];
cx q[4],q[3];
cx q[3],q[1];
cx q[3],q[4];
barrier q[0],q[1],q[2],q[3],q[4];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
