// EXPECTED_REWIRING [0 1 2 3 4 5 6 7 8 9 10 11 18 13 14 17 15 16 12 19]
// CURRENT_REWIRING [0 1 2 3 4 5 6 12 8 9 10 11 7 13 14 17 15 16 19 18]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[20];
rz(0.59368010174542*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.958108965734335*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.4741817804854853*pi) q[7];
rz(1.5707963267948966*pi) q[19];
rx(1.5707963267948966*pi) q[19];
cz q[18], q[19];
rz(0.10344064106915161*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.4189783790674746*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.3572636036508126*pi) q[18];
rx(-1.5707963267948966*pi) q[18];
rz(1.063603020062897*pi) q[18];
rx(-1.5707963267948966*pi) q[18];
cz q[18], q[12];
rz(1.6366529270088535*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.760407881182692*pi) q[18];
rx(-1.5707963267948966*pi) q[18];
cz q[18], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[18];
cz q[18], q[12];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[7];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[7];
rz(-1.1645820567151595*pi) q[18];
rx(1.5707963267948966*pi) q[18];
rz(0.1653856061068779*pi) q[18];
rx(-1.5707963267948966*pi) q[18];
rz(-0.5146654427613733*pi) q[18];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(-1.1645820567151592*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.16538560610687794*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(2.730367851897572*pi) q[12];
cz q[12], q[17];
rz(1.5707963267948966*pi) q[17];
rx(1.5707963267948966*pi) q[17];
cz q[17], q[18];
rz(2.6245865096257943*pi) q[19];
rx(1.5707963267948966*pi) q[19];
rz(1.384484161973147*pi) q[19];
rx(-1.5707963267948966*pi) q[19];
rz(-2.2762476260936904*pi) q[19];
rz(3.0381520125206416*pi) q[12];
cz q[11], q[18];
rz(-3.0381520125206416*pi) q[18];
rx(1.5707963267948966*pi) q[18];
rz(1.4189783790674746*pi) q[18];
rx(-1.5707963267948966*pi) q[18];
cz q[19], q[18];
rz(1.6366529270088535*pi) q[18];
rx(1.5707963267948966*pi) q[18];
rz(-1.5707963267948966*pi) q[19];
rx(-1.5707963267948966*pi) q[19];
cz q[19], q[18];
rx(-1.5707963267948966*pi) q[18];
rx(1.5707963267948966*pi) q[19];
cz q[19], q[18];
rz(-1.5707963267948966*pi) q[16];
rx(1.5707963267948966*pi) q[16];
cz q[16], q[14];
rz(-0.6542456812873576*pi) q[18];
rx(1.5707963267948966*pi) q[18];
rz(0.9242262418970197*pi) q[18];
rx(-1.5707963267948966*pi) q[18];
cz q[12], q[18];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(0.6463506146747173*pi) q[7];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(-1.5707963267948966*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
rz(3.141592653589793*pi) q[14];
rx(-1.5707963267948966*pi) q[16];
rz(1.5707963267948966*pi) q[16];
rx(-1.5707963267948966*pi) q[17];
rz(1.5707963267948966*pi) q[17];
rz(0.6463506146747173*pi) q[18];
rz(-1.1645820567151592*pi) q[19];
rx(1.5707963267948966*pi) q[19];
rz(0.16538560610687794*pi) q[19];
rx(-1.5707963267948966*pi) q[19];
rz(2.6269272108284194*pi) q[19];
