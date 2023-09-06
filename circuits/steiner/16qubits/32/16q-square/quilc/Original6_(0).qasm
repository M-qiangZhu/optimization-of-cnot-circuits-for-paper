// EXPECTED_REWIRING [0 1 2 4 3 6 9 8 5 7 10 11 12 13 14 15]
// CURRENT_REWIRING [5 15 3 4 0 6 7 13 10 1 9 2 12 11 8 14]
OPENQASM 2.0;
include "qelib1.inc";
qreg q[16];
rz(-1.5707963267948966*pi) q[4];
rx(1.5707963267948966*pi) q[4];
cz q[4], q[11];
cz q[4], q[5];
rz(-1.7843290499389812*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.077989633526896*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-0.8103884456122044*pi) q[13];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-2.5994471240966384*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(-0.5013159828795501*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-0.7648076667974766*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.799699369126487*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.6744345806143834*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.77356753144053*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-2.763416000932395*pi) q[13];
cz q[13], q[10];
rz(2.303086797794718*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rz(2.057108346694903*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.907934547781499*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(1.0597242096948425*pi) q[11];
rz(-2.1852696679121486*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.4044974404821478*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
cz q[11], q[12];
rx(-1.5707963267948966*pi) q[11];
rz(3.0866478387467406*pi) q[12];
rx(1.5707963267948966*pi) q[12];
cz q[11], q[12];
rx(1.5707963267948966*pi) q[11];
rx(-1.5707963267948966*pi) q[12];
cz q[11], q[12];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.6854761369280995*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.9010513749569045*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[9], q[10];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
cz q[9], q[6];
rz(0.59368010174542*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.958108965734335*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(-2.4741817804854853*pi) q[0];
rz(1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.7843290499389812*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.0779896335268964*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.381184772407101*pi) q[1];
cz q[1], q[0];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(1.0018103369216076*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.8023808521828162*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-0.6713901139333532*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.5787399662653339*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-2.0383130986546614*pi) q[6];
cz q[6], q[5];
rz(1.7126782741864437*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[5];
rz(1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[2];
rz(-0.01762479801721783*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.4675792633814504*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-2.4544692213837376*pi) q[5];
rz(0.24271325173162997*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.2615998376377684*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(0.6734730460682392*pi) q[0];
cz q[5], q[2];
rz(-1.7843290499389812*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(2.077989633526896*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(-0.8103884456122044*pi) q[3];
rx(-1.5707963267948966*pi) q[1];
rz(-0.9395111331491082*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.270896479992141*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(1.9548240462570623*pi) q[6];
cz q[6], q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.22770060095388445*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(-2.6625757902999436*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.8385954038498077*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(2.387104966695441*pi) q[1];
rz(-1.1645820567151615*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.16538560610687766*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[9];
rz(1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(0.002342968651979399*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.509545457964055*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.8674194433436226*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(0.40995421755535216*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
rz(0.26566153290242867*pi) q[11];
cz q[11], q[10];
rz(-1.070599534528215*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[11];
cz q[11], q[10];
rz(-2.57142409078096*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(2.701094802853649*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[9], q[10];
rz(3.141592653589793*pi) q[4];
rx(1.5707963267948966*pi) q[4];
rz(-0.5122687449609608*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.9870963507339323*pi) q[11];
rx(-1.5707963267948966*pi) q[11];
cz q[11], q[4];
rz(-2.298994492700352*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.0779896335268955*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(-2.3811847724071007*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(0.10344064106915161*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.4189783790674746*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[3], q[2];
rz(1.6366529270088535*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
cz q[3], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[3];
cz q[3], q[2];
rz(0.10344064106915161*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.4189783790674746*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-1.7843290499389812*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(2.0779896335268964*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.381184772407101*pi) q[9];
cz q[9], q[8];
rz(1.6366529270088535*pi) q[8];
rx(1.5707963267948966*pi) q[8];
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
rz(0.10344064106915161*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4189783790674746*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(-3.075736053375836*pi) q[5];
rz(-2.2756376828816034*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(-1.8629258098695765*pi) q[11];
rx(1.5707963267948966*pi) q[11];
cz q[10], q[11];
rz(0.4625730804621907*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(0.7579948982327972*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(0.15128825656490472*pi) q[12];
rz(0.055459446890653956*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.0403325567363917*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(-1.6440079963149596*pi) q[10];
rz(2.870653243632767*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.8031272074410487*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[10], q[13];
rx(-1.5707963267948966*pi) q[10];
rz(-0.6818876679986686*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[10], q[13];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
cz q[10], q[13];
cz q[0], q[7];
rz(-1.0136396841913253*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.843763984315056*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.48979898104064623*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.9414331880179325*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(3.099841298737381*pi) q[2];
cz q[2], q[1];
rz(-2.225485772181802*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(3.141592653589793*pi) q[2];
rx(1.5707963267948966*pi) q[2];
cz q[2], q[1];
rz(1.5525668733676397*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.7027105850489188*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[0], q[1];
rz(1.3572636036508112*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(2.077989633526896*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-0.8103884456122045*pi) q[7];
rz(-1.454700306421503*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.7351579045174503*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-2.2545234763835316*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(2.9939131596882693*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-1.7236490571387972*pi) q[6];
cz q[6], q[1];
rz(-0.6493001192343417*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(3.141592653589793*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(3.141592653589793*pi) q[6];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[9];
cz q[14], q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(2.920531050301563*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.5471227161787491*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(1.8541153414270812*pi) q[1];
rz(-1.1516645635418716*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.8489427393291833*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(-0.6542456812873562*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.9242262418970197*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-2.045495712376673*pi) q[9];
cz q[6], q[9];
rz(2.79528696812054*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.4189783790674746*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-3.075736053375836*pi) q[9];
rz(2.761369489712264*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(1.9641888827222767*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(-0.9438241621069082*pi) q[14];
rz(0.10344064106915161*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(1.4189783790674746*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
cz q[7], q[0];
rz(1.6366529270088535*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(-1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
cz q[7], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[0];
rz(-1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.5707963267948966*pi) q[15];
cz q[14], q[9];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(-1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
cz q[14], q[9];
rx(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[14];
cz q[14], q[9];
rz(-0.04812004727583721*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.8178457765615856*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(0.610125293381737*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.5716854502789275*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.1165140127480344*pi) q[10];
cz q[10], q[5];
rz(-2.5422887715575353*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[5];
rx(-1.5707963267948966*pi) q[5];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[5];
rz(-0.5263442591578993*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.0323661251041614*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-0.7250672780878857*pi) q[12];
rz(-0.8837720140269766*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.6586518466234943*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[12], q[13];
rx(-1.5707963267948966*pi) q[12];
rz(1.3640946752280758*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[12], q[13];
rz(3.141592653589793*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(3.141592653589793*pi) q[13];
rx(1.5707963267948966*pi) q[13];
cz q[12], q[13];
rz(-1.1645820567151588*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(0.16538560610687789*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(2.626927210828419*pi) q[14];
rz(0.8826858269124578*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(1.0636030200628976*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.760407881182692*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
cz q[6], q[1];
rx(-1.5707963267948966*pi) q[1];
rx(1.5707963267948966*pi) q[6];
cz q[6], q[1];
rz(-0.8842792068756182*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.0715623455425989*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-3.120242022050456*pi) q[2];
rz(-2.6625757902999436*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.8385954038498077*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(2.387104966695441*pi) q[0];
rz(2.4873469723024355*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(2.217366411692774*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[2];
rz(1.6287780582097862*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(2.145774494117323*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[14];
rz(-2.0234480977435063*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.1841253414612265*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(1.6801738619244821*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(2.018981305471792*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rz(-0.7502851010923163*pi) q[8];
cz q[8], q[7];
rz(1.3763424746260142*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(3.141592653589793*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[7];
rz(-2.0448784002780824*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.1459465616607365*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[15];
rz(-0.9037774224837134*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.7347109138658299*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.2940014354247684*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.5947388955923303*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.9742545804472957*pi) q[10];
cz q[10], q[9];
rz(-1.3172025463318193*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rx(-1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(3.141592653589793*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[10], q[9];
rz(3.141592653589793*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(1.5707963267948966*pi) q[11];
rz(-2.00361421832553*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(1.063603020062896*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(0.760407881182692*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
cz q[1], q[0];
rx(-1.5707963267948966*pi) q[0];
rx(1.5707963267948966*pi) q[1];
cz q[1], q[0];
rz(0.38531027190639344*pi) q[7];
rx(1.5707963267948966*pi) q[7];
rz(1.3719669514969317*pi) q[7];
rx(-1.5707963267948966*pi) q[7];
rz(-1.3113365952797222*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
cz q[8], q[7];
rx(-1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[14];
rx(1.5707963267948966*pi) q[14];
rz(-1.6083781207614667*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(2.7378437446369315*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[14], q[15];
rx(1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[15];
cz q[14], q[15];
rz(-2.229227014168261*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.5888634901089616*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(-1.8981744579753654*pi) q[9];
rz(-1.0610870553483402*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(0.8657220371091132*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(2.4298870976121236*pi) q[10];
cz q[13], q[10];
cz q[11], q[10];
rz(-0.008179932924801004*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.9222767004458257*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
rz(1.8995547394627872*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(0.8258573584729217*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(1.1207511613294683*pi) q[13];
cz q[13], q[10];
rz(-0.19731610918777376*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rx(-1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(-1.5707963267948966*pi) q[10];
rx(1.5707963267948966*pi) q[13];
cz q[13], q[10];
rx(1.5707963267948966*pi) q[14];
rz(0.40374890895286186*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(-2.368692159076627*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.924502630126994*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[9], q[10];
rz(2.7837311634860997*pi) q[13];
rx(1.5707963267948966*pi) q[13];
rz(1.7559048937174322*pi) q[13];
rx(-1.5707963267948966*pi) q[13];
rz(-1.6395171061957674*pi) q[13];
cz q[14], q[13];
rz(0.24271325173163072*pi) q[0];
rx(1.5707963267948966*pi) q[0];
rz(2.261599837637768*pi) q[0];
rx(-1.5707963267948966*pi) q[0];
rz(2.44514567110807*pi) q[7];
rx(1.5707963267948966*pi) q[7];
cz q[0], q[7];
rz(1.2286860578860317*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[15];
rz(0.39505811534853524*pi) q[15];
rx(-1.5707963267948966*pi) q[15];
cz q[8], q[15];
rz(-1.5707963267948966*pi) q[8];
rx(-1.5707963267948966*pi) q[8];
rx(1.5707963267948966*pi) q[15];
cz q[8], q[15];
rz(3.141592653589793*pi) q[8];
rx(1.5707963267948966*pi) q[8];
cz q[8], q[15];
rx(-1.5707963267948966*pi) q[6];
rz(1.5707963267948966*pi) q[6];
rz(-1.5707963267948966*pi) q[9];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[14];
rz(0.59368010174542*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(2.958108965734335*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
rz(-2.4741817804854853*pi) q[2];
cz q[10], q[11];
rz(0.21353272314408556*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(1.063603020062897*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.7604078811826924*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
cz q[9], q[6];
rx(-1.5707963267948966*pi) q[6];
rx(1.5707963267948966*pi) q[9];
cz q[9], q[6];
rz(1.5707963267948966*pi) q[14];
rx(-1.5707963267948966*pi) q[14];
rz(1.5707963267948966*pi) q[14];
rz(-1.3708191022537781*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4707173063589087*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.301745756457912*pi) q[5];
rz(-2.9891213948805735*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.1545017760970349*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[5], q[10];
rx(-1.5707963267948966*pi) q[5];
rz(1.5783032474507008*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[5], q[10];
rz(3.141592653589793*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(3.141592653589793*pi) q[10];
rx(1.5707963267948966*pi) q[10];
cz q[5], q[10];
rz(1.205020923005975*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(1.4520771477117036*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.2708884773668498*pi) q[5];
cz q[5], q[2];
rz(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(-0.6692762333944486*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
cz q[5], q[2];
rx(-1.5707963267948966*pi) q[2];
rx(1.5707963267948966*pi) q[5];
cz q[5], q[2];
rz(-1.1645820567151592*pi) q[3];
rx(1.5707963267948966*pi) q[3];
rz(0.16538560610687794*pi) q[3];
rx(-1.5707963267948966*pi) q[3];
rz(2.6269272108284194*pi) q[3];
rx(1.5707963267948966*pi) q[7];
cz q[7], q[8];
rz(-2.8988794018581636*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(0.8799928159520247*pi) q[2];
rx(-1.5707963267948966*pi) q[2];
cz q[2], q[3];
rz(-1.1645820567151592*pi) q[5];
rx(1.5707963267948966*pi) q[5];
rz(0.16538560610687794*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(2.730367851897572*pi) q[5];
rz(-1.6478338891432065*pi) q[10];
rx(1.5707963267948966*pi) q[10];
rz(1.1290013132633399*pi) q[10];
rx(-1.5707963267948966*pi) q[10];
cz q[5], q[10];
rz(-1.1645820567151592*pi) q[9];
rx(1.5707963267948966*pi) q[9];
rz(0.16538560610687794*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(2.730367851897572*pi) q[9];
cz q[9], q[14];
rz(-2.4681196075215537*pi) q[0];
rz(-1.1645820567151595*pi) q[1];
rx(1.5707963267948966*pi) q[1];
rz(0.1653856061068779*pi) q[1];
rx(-1.5707963267948966*pi) q[1];
rz(-0.5146654427613733*pi) q[1];
rz(0.8973232807266585*pi) q[2];
rx(1.5707963267948966*pi) q[2];
rz(1.5707963267948966*pi) q[2];
rz(-1.5707963267948966*pi) q[4];
rz(1.467355685725745*pi) q[5];
rx(-1.5707963267948966*pi) q[5];
rz(1.5707963267948966*pi) q[5];
rz(-0.6542456812873576*pi) q[6];
rx(1.5707963267948966*pi) q[6];
rz(0.9242262418970197*pi) q[6];
rx(-1.5707963267948966*pi) q[6];
rz(0.6463506146747164*pi) q[6];
rz(1.5707963267948966*pi) q[7];
rx(3.141592653589793*pi) q[7];
rz(0.3574763213819625*pi) q[8];
rx(1.5707963267948966*pi) q[8];
rz(1.5707963267948966*pi) q[8];
rz(1.467355685725745*pi) q[9];
rx(-1.5707963267948966*pi) q[9];
rz(1.5707963267948966*pi) q[9];
rz(-3.0310090767103937*pi) q[10];
rz(1.5707963267948966*pi) q[11];
rx(1.5707963267948966*pi) q[11];
rz(-1.5707963267948966*pi) q[11];
rz(-2.3929927398465822*pi) q[12];
rx(1.5707963267948966*pi) q[12];
rz(1.7372573266131044*pi) q[12];
rx(-1.5707963267948966*pi) q[12];
rz(-0.4582867482294614*pi) q[12];
rz(0.9867401414445904*pi) q[13];
rz(3.141592653589793*pi) q[14];
rz(-1.5707963267948966*pi) q[15];
rx(1.5707963267948966*pi) q[15];
rz(1.9129065957037608*pi) q[15];
