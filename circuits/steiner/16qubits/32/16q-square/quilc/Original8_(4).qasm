// EXPECTED_REWIRING [0 4 5 3 1 6 2 7 9 8 10 11 12 13 14 15]
// CURRENT_REWIRING [5 10 1 0 2 4 3 11 8 15 9 13 14 12 7 6]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(0.59368010174542*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.958108965734335*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.4741817804854853*pi) q[10];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[11];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(0.10344064106915161*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.4189783790674746*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-1.7843290499389812*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.0779896335268964*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-2.381184772407101*pi) q[12];
cz q[12], q[11];
rz(1.6366529270088535*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rz(0.59368010174542*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.958108965734335*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.4741817804854853*pi) q[9];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.1645820567151632*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.165385606106878*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-3.0381520125206416*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4189783790674746*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-3.075736053375836*pi) q[2];
cz q[3], q[4];
rz(0.59368010174542*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.958108965734335*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.4741817804854853*pi) q[8];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(0.10344064106915161*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4189783790674746*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(1.6366529270088535*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-1.1645820567151577*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.16538560610687805*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.4489977586567235*pi) q[8];
cz q[15], q[8];
rz(2.761369489712264*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.9641888827222767*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.9438241621069082*pi) q[9];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[0], q[7];
rz(-1.3524640940414476*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4702032720093634*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.5291443932617312*pi) q[2];
rz(1.4846948422348791*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.7679037741178978*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(-1.5707963267948966*pi) q[2];
rz(-1.9348629723243158*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(2.0377729569645013*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4189783790674746*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[9], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.1645820567151561*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.16538560610687786*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[13];
rz(-2.6625757902999436*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.8385954038498077*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.387104966695441*pi) q[8];
rz(1.8810846069711118*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.595101861999*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(2.227123434134475*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.5470639774872708*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-2.7284336413627246*pi) q[15];
cz q[15], q[14];
rz(0.05277177522259535*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(3.141592653589793*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(3.141592653589793*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(3.141592653589793*pi) q[13];
rz(0.02591178775616148*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.157760762010338*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.6667298309600993*pi) q[14];
rx(-1.5707963267948966*pi) q[9];
rz(-2.2989944927003583*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.0779896335268964*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(1.5707963267948966*pi) q[9];
rz(-2.381184772407101*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(1.4564375502462896*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4269954866939927*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(0.32781369042210384*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.6594069261530932*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.1885290783941111*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.4159359239908453*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-1.9660810788366323*pi) q[11];
cz q[11], q[4];
rz(0.18959471210626866*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-0.8923939263006844*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.7002121784908575*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.7376855417278563*pi) q[4];
rz(1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.7843290499389812*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.077989633526896*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.8103884456122044*pi) q[6];
rz(-0.6542456812873576*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.495242038915076*pi) q[9];
rz(-1.4888199713396473*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.4311265511855007*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.0959254239806964*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
cz q[9], q[10];
cz q[14], q[13];
rz(2.019185376763438*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.792997630704545*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-2.5122346208755784*pi) q[12];
rx(1.5707963267948966*pi) q[10];
cz q[13], q[10];
rz(0.10344064106915161*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-0.9244457121201792*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(1.4564375502462918*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4269954866939927*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[7];
rz(1.1899563194706428*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.043520823656659*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(2.4202955951757414*pi) q[15];
cz q[15], q[8];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-2.9026644838413858*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.495242038915076*pi) q[5];
rz(-1.4888199713396457*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
cz q[0], q[1];
cz q[6], q[5];
rz(-0.976946543301041*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.7270295694571451*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.33572772634458653*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.6070847171885825*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.199052806218842*pi) q[2];
cz q[2], q[1];
rz(-2.5960194577331155*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(0.9990654985946805*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.7171893011929129*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(0.7066512235505338*pi) q[2];
cz q[2], q[5];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[10];
rz(-2.547912551844372*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.958108965734335*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.474181780485484*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[10];
rz(-2.0451378530158433*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.5211976476655644*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.2795572863258995*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.6541453980492856*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(2.087802470758894*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.7571084916166462*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(0.8653450274961033*pi) q[9];
rz(-2.868467964585767*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.5951919787154443*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.815380020753187*pi) q[8];
cz q[9], q[8];
rz(-2.977447809153925*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.7843290499389812*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.077989633526896*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.8103884456122044*pi) q[14];
rz(-0.213532723144086*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.0779896335268955*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.381184772407101*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(0.3546334573839564*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.204524765469868*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[6];
rz(2.5479125518443735*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.18348368785545807*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.540038380699442*pi) q[0];
rz(0.7568768833477642*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.077989633526896*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-2.381184772407101*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.3572636036508101*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.077989633526894*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(-2.381184772407103*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[12];
rx(-1.5707963267948966*pi) q[9];
cz q[14], q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(2.6455078847184934*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.684375512795056*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.632116826285768*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.3533840010622376*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.5311349793634844*pi) q[6];
cz q[6], q[1];
rz(2.0640460075960725*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[5];
rz(-1.6095791669493675*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.8696548362862833*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.1310329540124735*pi) q[6];
cz q[6], q[5];
rx(1.5707963267948966*pi) q[5];
rz(-2.9120499743088497*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(1.5707963267948966*pi) q[7];
rz(-0.3072384032901745*pi) q[7];
rz(-0.6542456812873576*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.9242262418970197*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(0.6463506146747164*pi) q[12];
rz(-1.1645820567151592*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.16538560610687794*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(2.730367851897572*pi) q[13];
rz(1.4564375502462923*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.4269954866939927*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.652772682250146*pi) q[14];
cz q[13], q[14];
rz(0.24271325173162997*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.2615998376377684*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6734730460682391*pi) q[0];
rz(-1.1645820567151632*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.165385606106878*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[7];
rz(1.0561308840335282*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[6];
cz q[0], q[7];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[10], q[5];
rz(0.10344064106915161*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[9], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(1.6697967338704256*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.112864610827818*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.186283304745498*pi) q[1];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
cz q[6], q[1];
rz(1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.28067294034516516*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.356839106231828*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-1.1645820567151592*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687794*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.730367851897572*pi) q[9];
rz(1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-1.1645820567151557*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.16538560610687789*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
cz q[13], q[12];
rz(3.141592653589793*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(0.18781589801346144*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(2.845329369051865*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4876788469527376*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(3.1231845408577827*pi) q[3];
rz(0.2427132517316307*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.261599837637768*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.4681196075215537*pi) q[4];
rz(-2.495242038915076*pi) q[5];
rz(2.217146941469614*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.495242038915076*pi) q[8];
rz(3.0381520125206407*pi) q[9];
rz(-2.4870317424147474*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(0.2806729403451651*pi) q[11];
rx(3.141592653589793*pi) q[11];
rz(3.141592653589793*pi) q[12];
rz(1.467355685725745*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[14];
rz(1.0561308840335206*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(-1.5707963267948966*pi) q[15];
