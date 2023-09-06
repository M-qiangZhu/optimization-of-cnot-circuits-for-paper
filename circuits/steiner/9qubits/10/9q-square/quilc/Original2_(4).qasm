// EXPECTED_REWIRING [0 1 2 4 3 5 7 6 8]
// CURRENT_REWIRING [1 0 3 7 8 5 2 6 4]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[9];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[4];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.7843290499389812*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.077989633526896*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.8103884456122044*pi) q[8];
rz(1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[6];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(3.141592653589793*pi) q[6];
rz(0.10344064106915161*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[8], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[3];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.7936242567106415*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-1.0883480893858952*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[4], q[7];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[7];
cz q[4], q[7];
cz q[1], q[2];
rz(2.374273576535313*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.9821252657688615*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.6922336832463967*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.2304432606494144*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.681652918607383*pi) q[4];
cz q[4], q[3];
rz(0.6850662347270431*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-3.0381520125206416*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4189783790674746*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-3.075736053375836*pi) q[2];
rz(-2.854816445714298*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.4012143325178004*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(0.7872749811025557*pi) q[3];
cz q[3], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(1.6988253437119962*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.2581668809381504*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-0.39613663211897127*pi) q[0];
rz(-0.2169228284605479*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.8882443706221053*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(-1.5707963267948966*pi) q[0];
rz(-1.883521270435768*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(-0.12067704404556157*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.27420330249413927*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.3286367471971738*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.6499032820848805*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.8804342301276653*pi) q[4];
cz q[1], q[4];
rz(3.141592653589793*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-0.3479683968791525*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(-1.1645820567151592*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.16538560610687794*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.730367851897572*pi) q[3];
rz(-0.5011453761834481*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
cz q[5], q[6];
rx(-1.5707963267948966*pi) q[4];
cz q[7], q[4];
rz(-0.18601616321736936*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.1267276955110987*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-0.6135115018157493*pi) q[0];
rz(-2.7692372265607137*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(0.2427132517316307*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.261599837637768*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.4681196075215537*pi) q[2];
rz(-0.1034406410691524*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(3.141592653589793*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(-1.1645820567151592*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687794*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.6269272108284194*pi) q[8];
