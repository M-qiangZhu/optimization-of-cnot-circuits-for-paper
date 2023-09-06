// EXPECTED_REWIRING [0 1 2 3 4 5 7 6 8 9 10 11 13 12 14 15]
// CURRENT_REWIRING [10 9 11 2 12 0 6 13 7 8 1 3 5 4 15 14]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(0.10344064106915161*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-3.075736053375836*pi) q[1];
cz q[5], q[10];
rz(1.1384154740107841*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.942118687998284*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.9109514883197644*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.089477271439539*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.892929576718223*pi) q[6];
cz q[6], q[5];
rz(-3.002559241075489*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(-2.087802470758894*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.3844841619731474*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.2762476260936904*pi) q[9];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.5707963267948966*pi) q[0];
rz(2.8079966966946865*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.41543848027827707*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.2637533427426022*pi) q[6];
cz q[6], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-1.4597000595984486*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(0.24271325173163064*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.261599837637768*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.244269372863136*pi) q[1];
cz q[0], q[1];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[7], q[8];
rz(0.10344064106915161*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(1.053790182830899*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.3844841619731472*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.436141354291*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[9], q[6];
rx(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(2.761369489712264*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.9641888827222767*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.9438241621069082*pi) q[7];
rz(-0.6542456812873576*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.9242262418970197*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
cz q[0], q[1];
rz(1.468484313433314*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.0125331012365915*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.5271142084493476*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.227269400656863*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(1.0112026142548045*pi) q[3];
cz q[3], q[2];
rz(2.2021291074121425*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.7437907301540037*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.0496921951537208*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.0699974014672198*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.3088502052605655*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.4162233093670955*pi) q[10];
cz q[10], q[9];
rz(1.3917871387349376*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(3.141592653589793*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(0.7740060436714727*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.2783532062637788*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.815728329663653*pi) q[6];
rz(0.7086722564037171*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.5284868738287399*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rx(-1.5707963267948966*pi) q[6];
rz(2.9960338991142255*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[6], q[7];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rz(1.3849592185268063*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.5724540101793967*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.1651934448621555*pi) q[5];
rz(-1.4867615583333942*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.482120250874175*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[11];
rz(-2.8038956306707945*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.4189783790674746*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-1.7843290499389812*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.0779896335268964*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.381184772407101*pi) q[13];
cz q[13], q[10];
rz(1.6366529270088535*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(-1.9038013325814909*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.754423745298308*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(2.8039449583294878*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4416439879888072*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(2.608194203367244*pi) q[6];
rz(1.3775495739276882*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.642209555551563*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rx(-1.5707963267948966*pi) q[6];
rz(2.349044458710013*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[6], q[9];
rx(1.5707963267948966*pi) q[6];
rx(-1.5707963267948966*pi) q[9];
cz q[6], q[9];
rz(-3.1002478024823246*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.3707867215603846*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(0.20775051579996173*pi) q[9];
rz(3.0451498681382425*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.7733798711131373*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.7576233295448942*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(0.9107903645505832*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(1.5707963267948966*pi) q[5];
rz(2.4520146282984596*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-1.1826746570991549*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.7370835733564736*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.41848320726723237*pi) q[6];
cz q[6], q[1];
rz(1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-0.4892353831994205*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[1], q[6];
rz(-2.1644764285403126*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.1834836878554586*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(0.6674108731043042*pi) q[8];
rz(1.3635803380342881*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.519382134614907*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-1.3902738835853912*pi) q[10];
cz q[10], q[9];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-0.7533155441180073*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(0.59368010174542*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(2.958108965734335*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-2.4741817804854853*pi) q[4];
rz(-1.1645820567151592*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.16538560610687794*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.730367851897572*pi) q[10];
rz(-1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[10], q[11];
rz(0.35200090786716753*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.1825130093140213*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.442409601475319*pi) q[7];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[8];
rz(-2.898879401858163*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.8799928159520249*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(2.07559513449817*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.3543636222221993*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.8315256931336119*pi) q[3];
rz(3.141592653589793*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[12];
rz(-0.213532723144084*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(2.0779896335268955*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-2.381184772407101*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rx(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(1.9770105968746348*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.976207047482916*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(2.571871816572259*pi) q[13];
cz q[14], q[13];
rz(-2.244269372863135*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[15];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rz(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[3];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[3];
rx(-1.5707963267948966*pi) q[11];
rz(1.35726360365081*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(2.077989633526896*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-2.381184772407101*pi) q[12];
cz q[12], q[11];
rx(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
cz q[12], q[11];
rx(-1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[12];
cz q[12], q[11];
rz(-2.160647014880036*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.4189783790674746*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-3.075736053375836*pi) q[13];
rz(-0.5936801017454203*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(2.9581089657343353*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-1.7843290499389812*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.0779896335268964*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-2.381184772407101*pi) q[15];
cz q[15], q[14];
rz(2.238207199899205*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[15];
cz q[15], q[14];
rx(-1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[15];
cz q[15], q[14];
rz(1.3715820933206189*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(-0.6542456812873576*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.9242262418970197*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-2.495242038915076*pi) q[3];
rz(-2.891497888031587*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.4396030282326704*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(1.691408940458869*pi) q[8];
rz(0.6835342290603327*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.401981240134735*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[8], q[9];
rx(-1.5707963267948966*pi) q[8];
rz(-1.4166516979557873*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[8], q[9];
rx(1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[9];
cz q[8], q[9];
rz(1.467355685725745*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.5707963267948966*pi) q[10];
rz(-1.469390962039975*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.2666579117478176*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[10];
rz(-1.060211647322999*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.8350546282880786*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(3.024961656663736*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(3.041782815669968*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.6682413239706815*pi) q[14];
cz q[14], q[13];
rz(-1.9326387869388202*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-0.1301913360488327*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.3294912647772694*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(-1.9797521127171502*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.6847581620243268*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-0.12011524385234651*pi) q[4];
rz(1.4408633806565478*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.9851694034484986*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[4], q[11];
rz(1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-0.8080115540640627*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[4], q[11];
rz(2.4802317773285494*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.082026839302116*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.9226800534068387*pi) q[0];
rz(-2.825170554820463*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.6637809099492133*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(-1.5707963267948966*pi) q[0];
rz(-2.1738000623684908*pi) q[1];
rx(1.5707963267948966*pi) q[1];
cz q[0], q[1];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
cz q[6], q[5];
rz(-1.210310218280899*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.8967646834346055*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(3.141592653589793*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(1.452198725545379*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(1.285931465958436*pi) q[4];
cz q[3], q[4];
rz(-1.1645820567151592*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.16538560610687686*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(1.974120538792315*pi) q[12];
rz(1.9092591319969245*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.7865627446018654*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-2.4247210318812913*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.4421503684144259*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.4617408878156133*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(2.368015198528531*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-1.3534233131803888*pi) q[14];
cz q[14], q[9];
rz(-1.615693120151688*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(2.5711348369253635*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.4189783790674746*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.517006143963997*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.7571084916166453*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(1.6366529270088535*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-0.7054512992987938*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.2260506237081197*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4189783790674746*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-1.804939423768761*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.5466730560462866*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(1.6366529270088535*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.9613106845350554*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(1.4564375502462912*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.4269954866939933*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(3.1030101662908374*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.6932627659499886*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.030370457852639*pi) q[9];
cz q[6], q[9];
rz(-0.5936801017454197*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.18348368785545807*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-2.540038380699442*pi) q[7];
rz(3.1145736397737394*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.7860379100587506*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.5517008991600922*pi) q[8];
cz q[8], q[7];
rz(1.6366529270088535*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.2571766365573283*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[7];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(1.3705948070731353*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.18348368785545857*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(2.4309187827181695*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.809932073817397*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-1.7719086088343712*pi) q[14];
cz q[14], q[13];
rz(-0.9033854536905859*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(-2.873026795901799*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[13];
rx(-1.5707963267948966*pi) q[13];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[13];
rz(-1.1645820567151595*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(0.1653856061068779*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
rz(-0.5146654427613733*pi) q[15];
rz(-0.6542456812873576*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.9242262418970197*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.343163130880458*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.069331602076178*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(2.384183369086589*pi) q[2];
cz q[1], q[2];
rz(-0.6542456812873576*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(0.9242262418970197*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[6], q[7];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-1.1645820567151632*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.165385606106878*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[15];
rz(2.223602998831001*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(-0.6542456812873576*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.9242262418970197*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[12];
rz(-2.916894135123444*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.4442479578764598*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-0.9400278091819979*pi) q[0];
rz(2.217146941469614*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.5707963267948966*pi) q[1];
rz(3.141592653589793*pi) q[2];
rz(1.4564375502462918*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(1.426995486693993*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-3.059616298134544*pi) q[3];
rz(1.8556611876313571*pi) q[4];
rx(-1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[4];
rz(-1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-1.4888199713396468*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(0.6463506146747173*pi) q[7];
rz(-1.1645820567151595*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(0.1653856061068779*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.5146654427613733*pi) q[8];
rz(-2.286179276489876*pi) q[9];
rz(-2.847947957019482*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[12];
rz(2.217146941469614*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.5707963267948966*pi) q[13];
rz(1.0561308840335282*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(-1.5707963267948966*pi) q[14];
rz(3.141592653589793*pi) q[15];
