// EXPECTED_REWIRING [2 14 3 1 4 5 6 7 15 9 10 11 13 12 8 0]
// CURRENT_REWIRING [4 5 14 10 7 9 0 15 6 12 8 11 3 13 2 1]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.2802944030552141*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.9035101846364597*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-2.087802470758894*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.3844841619731474*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-2.2762476260936904*pi) q[11];
rz(0.10344064106915161*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-3.075736053375836*pi) q[0];
rz(1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-2.355299346725589*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-1.0542536091141224*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[13], q[14];
rz(-1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(3.141592653589793*pi) q[14];
rx(1.5707963267948966*pi) q[14];
cz q[13], q[14];
rz(3.141592653589793*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[14];
rz(1.7920793693212373*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.7194151849644502*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.345469899593271*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.5006056868355997*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(0.25174904941748144*pi) q[15];
cz q[15], q[14];
rz(-1.7508048587398113*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rx(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(2.238991111098907*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.831833082080827*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-1.8868070875089922*pi) q[15];
cz q[15], q[0];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.2107590553795742*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[0];
rz(-1.7843290499389812*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.077989633526896*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.8103884456122044*pi) q[2];
rz(1.805996176575181*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.5617081027197315*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(2.2616339797736265*pi) q[0];
rz(-0.4200699612364365*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(-1.5707963267948966*pi) q[0];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(1.5707963267948966*pi) q[0];
cz q[0], q[1];
rz(-1.1645820567151595*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.1653856061068779*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.5146654427613733*pi) q[15];
cz q[0], q[15];
rz(1.7798094751098716*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.8166616465355656*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[2], q[1];
rz(-2.9315207165987704*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(1.1061857292730761*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.327887076922867*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(2.7551865416576575*pi) q[14];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(-0.9464158666467356*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9712684004281071*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.5362333019804044*pi) q[0];
rz(2.7261279146775435*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.9015711497020742*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[0], q[15];
rx(-1.5707963267948966*pi) q[0];
rz(0.985402688152397*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[0], q[15];
rz(3.141592653589793*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(3.141592653589793*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[0], q[15];
rz(-0.5255831582625585*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.6241446874684226*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.906737417783343*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.7999867567419805*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.759067312274805*pi) q[1];
cz q[1], q[0];
rz(1.599729598434255*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-3.06481360972428*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.6512652121730808*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.33213429702056*pi) q[2];
rz(2.1710343178300384*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rx(-1.5707963267948966*pi) q[2];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[3];
rz(2.775921060443034*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.36652883362188515*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.6041686701923408*pi) q[0];
rz(-2.087802470758894*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.3844841619731474*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.2762476260936904*pi) q[7];
rz(0.10344064106915161*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4189783790674746*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-3.075736053375836*pi) q[6];
cz q[7], q[0];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-2.6625757902999436*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.8385954038498077*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(2.387104966695441*pi) q[0];
rz(1.2674895377155633*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.6610750846628325*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.0990569195063539*pi) q[1];
rz(-1.5348675726482315*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.6047787480241893*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rx(-1.5707963267948966*pi) q[1];
rz(3.032942411140219*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[1], q[2];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(-0.21322002074600555*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.107412255447956*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-0.33441622392491865*pi) q[2];
rz(-1.4971614027604505*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.8902799075760388*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(0.8769235960025236*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.0544571090214339*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.6170409424437739*pi) q[4];
cz q[4], q[3];
rz(-0.9884571883659996*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.5707963267948966*pi) q[12];
rz(1.2058677941936566*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.284022034032548*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.27190214859824846*pi) q[14];
rz(1.8347324222347858*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.13290451981682644*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[14], q[15];
rx(-1.5707963267948966*pi) q[14];
rz(-2.377182488416519*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[14], q[15];
rz(3.141592653589793*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(3.141592653589793*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[14], q[15];
rz(0.6697504701118948*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.1103822270208044*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(0.47064264403688627*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.8635099471961216*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.5388390442575872*pi) q[4];
rz(-2.6432651850947044*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.2487946392552798*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.4599884759290664*pi) q[0];
rz(-2.833602635378541*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.6587040590333764*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(-1.5707963267948966*pi) q[0];
rz(-0.9770139364007138*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(1.0369596993975263*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.463117604933908*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.8075216618327841*pi) q[15];
rz(-0.6076103220911448*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.767260907446226*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.5647509535587258*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.2203114383705131*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.27312050432006646*pi) q[7];
cz q[7], q[6];
rz(-2.2523188598250456*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(3.141592653589793*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-2.2471990892610543*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.21834910836462956*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.364391144512913*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(-0.6884165991465693*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.2403554304630099*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.7027653823360114*pi) q[10];
cz q[11], q[10];
rz(-3.1043531925146493*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(3.0644552695251486*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.18348368785545793*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.5400383806994427*pi) q[13];
rz(-0.41387617297769014*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.0422198602236155*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.7887907384947317*pi) q[14];
cz q[14], q[13];
rz(1.6366529270088535*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-1.7954433759802055*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-0.8709421442322886*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.4189783790674746*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[4], q[3];
rz(1.6366529270088535*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(-1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(2.7900462930119954*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.1496225729018854*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(0.45782860826495986*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.1902914146491357*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-0.6470255028556549*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.2809526102868705*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(0.44205631635000214*pi) q[15];
cz q[15], q[0];
rz(-1.1717471315333556*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[0];
rz(-1.460959820728538*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.6559665427240893*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-3.073594846821393*pi) q[0];
rz(-2.9458628782659075*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.1552335457373148*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.7224760632617726*pi) q[1];
rz(-0.2970801473633564*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-1.925591051023322*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.1683337301122134*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[12], q[13];
rx(1.5707963267948966*pi) q[12];
rz(2.2616521440004744*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[12], q[13];
rz(0.8128445487985332*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.121736364639848*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(1.140790802801481*pi) q[15];
cz q[15], q[14];
rz(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.578679476811967*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rx(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
cz q[1], q[0];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
cz q[8], q[7];
rz(0.5609488992491204*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.9049036232169179*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.0728061214328415*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.6766117605600166*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.8796661831942938*pi) q[7];
cz q[7], q[0];
rz(2.9913673301530954*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(1.977010596874633*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.9762070474829163*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[2];
rz(2.844349627804023*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.3134202835792745*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(2.085461769556269*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-0.3433054312260424*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-2.345432100007103*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.634160414488931*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(2.988724332905952*pi) q[15];
cz q[15], q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[8];
rz(-1.1645820567151592*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.16538560610687794*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(2.730367851897572*pi) q[4];
cz q[4], q[5];
rz(-2.151878313673433*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(-0.11952709161080938*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.5760722258692728*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.0995365257563248*pi) q[7];
cz q[6], q[7];
rz(1.467355685725745*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.5707963267948966*pi) q[4];
rz(0.10344064106915161*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.9280599304457082*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.0779896335268955*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.6366529270088535*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-2.3811847724071025*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.2830391709598328*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.1799538750950662*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.8814952950542578*pi) q[0];
rz(-0.4663057838508547*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.5161109324732136*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[0], q[15];
rx(-1.5707963267948966*pi) q[0];
rz(-2.0676346433847694*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[0], q[15];
rz(3.141592653589793*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(3.141592653589793*pi) q[15];
rx(1.5707963267948966*pi) q[15];
cz q[0], q[15];
rz(-2.6581760512250683*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.077989633526896*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.810388445612205*pi) q[7];
rz(3.0519885164732896*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.0357052158473499*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.5738593939969026*pi) q[0];
rz(1.2541253058002573*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.565279434692574*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(-1.5707963267948966*pi) q[0];
rz(2.3664441325416856*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(3.141592653589793*pi) q[2];
rz(2.4608469202221976*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.7557804130594825*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-1.7186737039066484*pi) q[0];
cz q[7], q[0];
rz(-1.7917435660982335*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-0.23908064765095882*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.938158292487462*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.2911311730982627*pi) q[1];
rz(0.9038315646703685*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[1], q[2];
rz(0.08599843437246031*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.3451425612375347*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.5684542492954764*pi) q[0];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.1598449493429825*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(1.7703610580515348*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.173054023033306*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(0.9154467440687262*pi) q[7];
rz(0.4385328641530504*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.7244218501214787*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[7], q[8];
rx(-1.5707963267948966*pi) q[7];
rz(2.0935476459334463*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[7], q[8];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[7], q[8];
rz(0.24271325173162997*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.2615998376377684*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.8973232807266575*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[6];
rz(0.7829223353601898*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.462111195949934*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.2295165363329024*pi) q[7];
cz q[7], q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.13424549821057505*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(-3.0381520125206416*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.4189783790674746*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-3.075736053375836*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(1.5707963267948966*pi) q[0];
rz(0.31478750601002514*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-1.1645820567151592*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687794*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.730367851897572*pi) q[7];
cz q[0], q[7];
rz(-1.790637814553106*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.735240130905459*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(3.1271613816871713*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.9120286872278855*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(0.9083000445471533*pi) q[1];
cz q[1], q[0];
rz(2.911414910715794*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(0.3238533328050849*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.6061222692652863*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(2.830259375291205*pi) q[8];
rz(-2.435320908634976*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.341255748646676*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-1.674236967864049*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-0.8190509922251786*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.41420040160971*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-3.0552871225152702*pi) q[9];
rz(1.674236967864048*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.4189783790674746*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[8], q[7];
rz(1.6366529270088535*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(2.0180116704300834*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.324557238185653*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.4330285832676175*pi) q[15];
rx(-1.5707963267948966*pi) q[8];
cz q[9], q[8];
rx(1.5707963267948966*pi) q[8];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-0.6542456812873576*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.9242262418970197*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-2.495242038915076*pi) q[10];
rz(-1.6851551033435002*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.7145971668958007*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.3277114290435508*pi) q[9];
cz q[10], q[9];
rz(-3.0557344197174867*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.451597079108576*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-1.3726057375855838*pi) q[11];
rz(1.8938872912738156*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.271396736867185*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[11], q[12];
rx(-1.5707963267948966*pi) q[11];
rz(-1.2968074762534805*pi) q[12];
rx(1.5707963267948966*pi) q[12];
cz q[11], q[12];
rx(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
cz q[11], q[12];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.495242038915076*pi) q[7];
rz(-0.6542456812873576*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.9242262418970197*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[15];
rz(2.217146941469614*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
cz q[8], q[7];
rz(-0.21353272314408464*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.077989633526896*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.3811847724071016*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[4];
rz(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(-1.0781638657356778*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.3993544474049404*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-2.4022043684153656*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.8393465260865843*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(0.08008029654960058*pi) q[9];
cz q[9], q[8];
rz(2.0926390336357716*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(1.1451406131681259*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.958108965734335*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6015542728903507*pi) q[0];
rz(-1.7843290499389812*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.0779896335268964*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-2.381184772407101*pi) q[15];
cz q[15], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[0];
rz(-0.7330456707385032*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.511228481419105*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.9770105968746354*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.976207047482915*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(0.6918664355509236*pi) q[15];
cz q[8], q[15];
rz(-0.9038960953734599*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.146694483481283*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(3.0232007660997473*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.7779899747225123*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(2.345101097936194*pi) q[11];
cz q[11], q[10];
rz(-2.3506378790430116*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(3.141592653589793*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[5];
rz(1.3572636036508126*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.0636030200628968*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.760407881182692*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(2.6245865096257956*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.3844841619731472*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.2762476260936904*pi) q[7];
rx(-1.5707963267948966*pi) q[6];
cz q[7], q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[6];
rz(0.17176270045062245*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.0600369760704087*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.8702344081806792*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.6459926699126002*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-0.7133686019880905*pi) q[10];
cz q[9], q[10];
rz(0.7844539308867926*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.6730429820502648*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.5295377808282629*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.077989633526896*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(0.0815552353256237*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(-2.381184772407102*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.1645820567151592*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.16538560610687794*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(2.730367851897572*pi) q[7];
rz(0.24271325173162997*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.2615998376377684*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.8973232807266575*pi) q[8];
cz q[8], q[7];
rx(1.5707963267948966*pi) q[2];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[2], q[3];
rz(1.1742703354119124*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.217146941469614*pi) q[3];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(3.141592653589793*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.2737161794315395*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.6542456812873576*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.9242262418970197*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.217146941469614*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[13], q[14];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6463506146747173*pi) q[0];
rz(-1.360699375263867*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.7475890714341145*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.8014732080622837*pi) q[1];
rz(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
rz(1.5707963267948966*pi) q[3];
rz(-0.6542456812873576*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(0.9242262418970197*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(0.6463506146747164*pi) q[4];
rz(-0.6542456812873576*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.9242262418970197*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.6463506146747164*pi) q[5];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.495242038915076*pi) q[6];
rz(-0.1034406410691524*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(3.039007402358455*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4896679642684711*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.4265176356486704*pi) q[9];
rz(-0.8899640543140617*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(-1.015562830227508*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.6566824855509114*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(-2.9258431016106563*pi) q[11];
rz(-1.897494556011516*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.9937034781652719*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.7284450711310777*pi) q[12];
rx(-1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(-1.5707963267948966*pi) q[14];
rz(1.393595334005348*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
