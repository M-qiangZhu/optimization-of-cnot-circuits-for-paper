// EXPECTED_REWIRING [6 2 1 3 4 0 5 7 8]
// CURRENT_REWIRING [6 1 2 3 4 5 0 8 7]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[7];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[8], q[3];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
cz q[0], q[1];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(0.10344064106915161*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4189783790674746*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.784329049938982*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.0636030200628972*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(1.6366529270088535*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.760407881182692*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[0];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(-1.5707963267948966*pi) q[0];
rz(2.509361221683177*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-2.4694852659377577*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(-1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[2];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[4], q[7];
cz q[1], q[0];
rx(1.5707963267948966*pi) q[4];
cz q[5], q[4];
rz(0.10344064106915161*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.6245865096257965*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.7571084916166466*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-0.705451299298793*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[0];
rz(2.217146941469613*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-1.1645820567151586*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.16538560610687772*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.495242038915076*pi) q[0];
rz(2.4694852659377577*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-2.509361221683176*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[4];
rx(3.141592653589793*pi) q[4];
rz(-0.5146654427613746*pi) q[5];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(-1.1645820567151592*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687794*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.6269272108284194*pi) q[8];
