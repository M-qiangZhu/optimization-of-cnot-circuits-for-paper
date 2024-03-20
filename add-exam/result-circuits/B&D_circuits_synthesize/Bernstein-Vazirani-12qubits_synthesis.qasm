OPENQASM 2.0;
include "qelib1.inc";
qreg q[12];
creg meas[12];
cx q[11],q[9];
cx q[9],q[10];
cx q[10],q[8];
cx q[8],q[6];
cx q[6],q[5];
cx q[5],q[4];
cx q[4],q[2];
cx q[9],q[11];
cx q[10],q[9];
cx q[8],q[10];
cx q[6],q[8];
cx q[5],q[6];
cx q[4],q[5];
cx q[2],q[4];
cx q[0],q[2];
cx q[1],q[2];
cx q[4],q[2];
cx q[2],q[4];
cx q[4],q[2];
cx q[4],q[5];
cx q[5],q[6];
cx q[5],q[4];
cx q[6],q[5];
cx q[3],q[6];
cx q[6],q[5];
cx q[5],q[6];
cx q[4],q[5];
cx q[6],q[5];
cx q[5],q[6];
cx q[6],q[5];
cx q[8],q[6];
cx q[6],q[8];
cx q[8],q[6];
cx q[7],q[8];
cx q[10],q[8];
cx q[8],q[10];
cx q[10],q[8];
cx q[9],q[10];
cx q[10],q[9];
cx q[9],q[10];
cx q[11],q[9];
cx q[9],q[11];
cx q[11],q[9];
barrier q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8],q[9],q[10],q[11];
measure q[0] -> meas[0];
measure q[1] -> meas[1];
measure q[2] -> meas[2];
measure q[3] -> meas[3];
measure q[4] -> meas[4];
measure q[5] -> meas[5];
measure q[6] -> meas[6];
measure q[7] -> meas[7];
measure q[8] -> meas[8];
measure q[9] -> meas[9];
measure q[10] -> meas[10];
measure q[11] -> meas[11];