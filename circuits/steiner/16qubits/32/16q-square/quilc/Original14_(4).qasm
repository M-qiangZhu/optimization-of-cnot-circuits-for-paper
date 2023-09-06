// EXPECTED_REWIRING [0 1 2 3 4 5 6 7 8 9 13 11 12 10 14 15]
// CURRENT_REWIRING [0 6 2 3 4 7 11 8 14 15 13 5 1 12 10 9]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(0.59368010174542*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.958108965734335*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.4741817804854853*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-0.11684953538138552*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9847061932198287*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.7692201347043124*pi) q[5];
rz(2.7938757589357204*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.7850522907831803*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rx(-1.5707963267948966*pi) q[5];
rz(-1.368011358870414*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[5], q[6];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[5], q[6];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(1.3663200984099744*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.6313929692514453*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[10], q[11];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[11];
cz q[10], q[11];
rz(0.10072861736596206*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.8327812548453912*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[9], q[6];
rz(-1.3149601816263687*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.1641669332654447*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-3.042135748498398*pi) q[9];
rz(-1.071107554346602*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.679491563241163*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[9], q[14];
rx(-1.5707963267948966*pi) q[9];
rz(-2.5039966382410626*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[9], q[14];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[9], q[14];
rz(-3.0188724275335472*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5874869984047466*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.5687377974128363*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.6216912931147656*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(0.0306039309647156*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.958108965734335*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.6015542728903505*pi) q[6];
rz(-0.06905571780113379*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.2166371015456938*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.594778522312254*pi) q[9];
cz q[9], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.0951593596566367*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-1.1645820567151592*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687794*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.730367851897572*pi) q[9];
cz q[9], q[8];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.6313929692514442*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[10], q[13];
rz(0.10344064106915161*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4189783790674746*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.7843290499389812*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.0779896335268964*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-2.381184772407101*pi) q[15];
cz q[15], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[6];
rz(1.2538229625816601*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.0779896335268964*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.3811847724071016*pi) q[9];
cz q[9], q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(0.10344064106915161*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4189783790674746*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-3.075736053375836*pi) q[2];
rz(-2.3918013978459243*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.928059930445716*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.0636030200628972*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.7604078811826915*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-0.654245681287358*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.164582056715157*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687864*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.553777878862958*pi) q[9];
cz q[8], q[9];
rz(-1.7808890500813233*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.9253981187504015*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.9646505028407368*pi) q[14];
rz(0.10344064106915161*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
cz q[9], q[14];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.495242038915076*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-1.1645820567151592*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687794*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.730367851897572*pi) q[6];
cz q[9], q[6];
rz(-1.853161646075536*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.685190844190743*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-2.3232292535335675*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.4877748840164005*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-1.4419790024579884*pi) q[15];
cz q[15], q[14];
rz(-2.0805267487872197*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rx(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(-0.40113382584624935*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.178000296541483*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.0785041912113424*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.545026407281064*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.1477914400169333*pi) q[2];
cz q[2], q[1];
rz(0.23362339419857747*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-0.869838649084334*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.2159195906933558*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(0.10344064106915161*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4189783790674746*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.003614218325528*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.0779896335268964*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(1.6366529270088535*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-2.381184772407101*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-1.8752201369418549*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.8924742254989672*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[9], q[14];
rz(-0.4346941645636231*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.3574303033192028*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(0.545486368821332*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.485344558121403*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.7051668598574524*pi) q[10];
cz q[10], q[9];
rz(-2.7656297422449914*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(1.5707963267948966*pi) q[11];
rz(2.937116425204872*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[12];
rz(1.2892785507327853*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.871175192704527*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.503800010835234*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(0.23125817793189485*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.6202896950454084*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-2.657802785158222*pi) q[15];
rz(-1.1645820567151595*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.1653856061068779*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.5146654427613733*pi) q[8];
rz(1.8834027765540684*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.9604179355542464*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.9653460991632714*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.8183672103638363*pi) q[10];
cz q[11], q[10];
rz(2.0693722314433245*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.5293583611431096*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-0.5825210767969661*pi) q[11];
rz(0.8453355365207572*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.0330703665774887*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[11], q[12];
rx(-1.5707963267948966*pi) q[11];
rz(0.5007358558852619*pi) q[12];
rx(1.5707963267948966*pi) q[12];
cz q[11], q[12];
rx(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
cz q[11], q[12];
rz(1.9770105968746343*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.976207047482915*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.0482160593382028*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-1.2598696414559136*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.2344160494273377*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.369584124472636*pi) q[11];
cz q[11], q[10];
rz(1.6366529270088535*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.3065097237552266*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(2.547912551844372*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.18348368785545782*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-0.903385453690588*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.24271325173163075*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.2615998376377675*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(1.5707963267948966*pi) q[0];
rx(3.141592653589793*pi) q[0];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(1.6366529270088535*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(0.0723857128496821*pi) q[1];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[7], q[6];
rz(-0.9244457121201792*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-0.0695312004859101*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-1.2797216771077478*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.7544761786317151*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rx(1.5707963267948966*pi) q[1];
rz(-0.0225471547710062*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(-0.5391247400695257*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.7571084916166455*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.8653450274961029*pi) q[5];
cz q[7], q[0];
rz(-0.23050605567031254*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.4181779075498064*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.838169512091148*pi) q[1];
rz(-0.21353272314408464*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.077989633526896*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.3811847724071016*pi) q[6];
cz q[6], q[1];
rz(0.19515877187647312*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-3.0491373132796404*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.0218957517737746*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.7345312404281406*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[4], q[11];
rx(1.5707963267948966*pi) q[4];
rz(-1.5473829865252826*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[4], q[11];
rz(-1.3593184030916068*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.8958981354074986*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(3.0787740059932758*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.463404573139461*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-1.1843175970400968*pi) q[5];
cz q[5], q[2];
rz(-1.0980228883862573*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-3.0352890066903995*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.4479179797085187*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.2608103886338893*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.477422494952751*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-1.2079074948524227*pi) q[3];
cz q[3], q[2];
rz(-2.6202080359619604*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(0.953217866425792*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.484198391506175*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.14107813280188963*pi) q[6];
rz(-0.8973232807266585*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-0.9628142232258217*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(2.761369489712264*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.9641888827222767*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-0.9438241621069082*pi) q[9];
rz(2.217146941469614*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.4347497409174617*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.7908536431482283*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(-0.17302493717264777*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.0689971562572516*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.7909553869918127*pi) q[6];
cz q[9], q[6];
rz(0.33317289718642984*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-0.15716892762338786*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.5781727094058032*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.43164952259688594*pi) q[4];
rz(-1.4367820112111906*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.769987953578363*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rx(-1.5707963267948966*pi) q[4];
rz(-3.1221503705833102*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[4], q[5];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[4], q[5];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[1], q[6];
rz(0.30563133856412145*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.8764189729613028*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.722292999226706*pi) q[4];
rz(-1.5954407199347864*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4542242138071613*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[6];
rz(-2.3429320919709005*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.077989633526896*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.3811847724071016*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-2.495242038915076*pi) q[6];
rz(-1.2656527252067884*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.7223903671883827*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4101795901162901*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9595294413404539*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.473585832782433*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[3];
rz(1.7454667090538436*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(-1.1645820567151632*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.165385606106878*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[6];
rz(1.4564375502462914*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4269954866939922*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(0.09186809389598981*pi) q[9];
rz(-1.1645820567151557*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.16538560610687744*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
cz q[14], q[15];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.15159404039348523*pi) q[3];
rz(0.2427132517316307*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.261599837637768*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.4681196075215537*pi) q[4];
rz(1.0561308840335282*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-1.5707963267948966*pi) q[5];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[7];
rz(0.5030047909171906*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.5707963267948966*pi) q[7];
rz(3.141592653589793*pi) q[8];
rz(1.5609045883541555*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[9];
rz(2.6269272108284163*pi) q[10];
rz(3.141592653589793*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.6632516671050488*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-1.5707963267948966*pi) q[11];
rz(-1.8935088840833902*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.9598342347841504*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-2.206879782878821*pi) q[12];
rx(1.5707963267948966*pi) q[13];
rz(-1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(3.141592653589793*pi) q[15];
