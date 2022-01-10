

#include <vector>
#include <complex>
#include <stdio.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <iomanip>
#include <stdlib.h>
#include <string.h>
#include <gsl/gsl_siman.h>


using namespace std;

// Gauss-Legendre setup: you may choose from one of the following scheme:
// number of nodes = 64 costs around 0.25 sec to opt = 1e-12 <Recommended.>
// can ensure 1e-8 accuracy for the option price. that is, pvtrapz - pvgauss <=1e-8.
static double x64[32] = {0.0243502926634244325089558,0.0729931217877990394495429,0.1214628192961205544703765,0.1696444204239928180373136,0.2174236437400070841496487,0.2646871622087674163739642,0.3113228719902109561575127,0.3572201583376681159504426,0.4022701579639916036957668,0.4463660172534640879849477,0.4894031457070529574785263,0.5312794640198945456580139,0.5718956462026340342838781,0.6111553551723932502488530,0.6489654712546573398577612,0.6852363130542332425635584,0.7198818501716108268489402,0.7528199072605318966118638,0.7839723589433414076102205,0.8132653151227975597419233,0.8406292962525803627516915,0.8659993981540928197607834,0.8893154459951141058534040,0.9105221370785028057563807,0.9295691721319395758214902,0.9464113748584028160624815,0.9610087996520537189186141,0.9733268277899109637418535,0.9833362538846259569312993,0.9910133714767443207393824,0.9963401167719552793469245,0.9993050417357721394569056};
static double w64[32] = {0.0486909570091397203833654,0.0485754674415034269347991,0.0483447622348029571697695,0.0479993885964583077281262,0.0475401657148303086622822,0.0469681828162100173253263,0.0462847965813144172959532,0.0454916279274181444797710,0.0445905581637565630601347,0.0435837245293234533768279,0.0424735151236535890073398,0.0412625632426235286101563,0.0399537411327203413866569,0.0385501531786156291289625,0.0370551285402400460404151,0.0354722132568823838106931,0.0338051618371416093915655,0.0320579283548515535854675,0.0302346570724024788679741,0.0283396726142594832275113,0.0263774697150546586716918,0.0243527025687108733381776,0.0222701738083832541592983,0.0201348231535302093723403,0.0179517157756973430850453,0.0157260304760247193219660,0.0134630478967186425980608,0.0111681394601311288185905,0.0088467598263639477230309,0.0065044579689783628561174,0.0041470332605624676352875,0.0017832807216964329472961};

// number of nodes = 96 costs around 1.1 sec to opt = 1e-12
//static double x96[48] = {0.0162767448496029695791346,0.0488129851360497311119582,0.0812974954644255589944713,0.1136958501106659209112081,0.1459737146548969419891073,0.1780968823676186027594026,0.2100313104605672036028472,0.2417431561638400123279319,0.2731988125910491414872722,0.3043649443544963530239298,0.3352085228926254226163256,0.3656968614723136350308956,0.3957976498289086032850002,0.4254789884073005453648192,0.4547094221677430086356761,0.4834579739205963597684056,0.5116941771546676735855097,0.5393881083243574362268026,0.5665104185613971684042502,0.5930323647775720806835558,0.6189258401254685703863693,0.6441634037849671067984124,0.6687183100439161539525572,0.6925645366421715613442458,0.7156768123489676262251441,0.7380306437444001328511657,0.7596023411766474987029704,0.7803690438674332176036045,0.8003087441391408172287961,0.8194003107379316755389996,0.8376235112281871214943028,0.8549590334346014554627870,0.8713885059092965028737748,0.8868945174024204160568774,0.9014606353158523413192327,0.9150714231208980742058845,0.9277124567223086909646905,0.9393703397527552169318574,0.9500327177844376357560989,0.9596882914487425393000680,0.9683268284632642121736594,0.9759391745851364664526010,0.9825172635630146774470458,0.9880541263296237994807628,0.9925439003237626245718923,0.9959818429872092906503991,0.9983643758631816777241494,0.9996895038832307668276901};
//static double w96[48] = {0.0325506144923631662419614,0.0325161187138688359872055,0.0324471637140642693640128,0.0323438225685759284287748,0.0322062047940302506686671,0.0320344562319926632181390,0.0318287588944110065347537,0.0315893307707271685580207,0.0313164255968613558127843,0.0310103325863138374232498,0.0306713761236691490142288,0.0302999154208275937940888,0.0298963441363283859843881,0.0294610899581679059704363,0.0289946141505552365426788,0.0284974110650853856455995,0.0279700076168483344398186,0.0274129627260292428234211,0.0268268667255917621980567,0.0262123407356724139134580,0.0255700360053493614987972,0.0249006332224836102883822,0.0242048417923646912822673,0.0234833990859262198422359,0.0227370696583293740013478,0.0219666444387443491947564,0.0211729398921912989876739,0.0203567971543333245952452,0.0195190811401450224100852,0.0186606796274114673851568,0.0177825023160452608376142,0.0168854798642451724504775,0.0159705629025622913806165,0.0150387210269949380058763,0.0140909417723148609158616,0.0131282295669615726370637,0.0121516046710883196351814,0.0111621020998384985912133,0.0101607705350084157575876,0.0091486712307833866325846,0.0081268769256987592173824,0.0070964707911538652691442,0.0060585455042359616833167,0.0050142027429275176924702,0.0039645543384446866737334,0.0029107318179349464084106,0.0018539607889469217323359,0.0007967920655520124294381};

// number of nodes = 128 costs around 1.5 sec to opt = 1e-12
//static double x128[64] = {0.0122236989606157641980521,0.0366637909687334933302153,0.0610819696041395681037870,0.0854636405045154986364980,0.1097942311276437466729747,0.1340591994611877851175753,0.1582440427142249339974755,0.1823343059853371824103826,0.2063155909020792171540580,0.2301735642266599864109866,0.2538939664226943208556180,0.2774626201779044028062316,0.3008654388776772026671541,0.3240884350244133751832523,0.3471177285976355084261628,0.3699395553498590266165917,0.3925402750332674427356482,0.4149063795522750154922739,0.4370245010371041629370429,0.4588814198335521954490891,0.4804640724041720258582757,0.5017595591361444642896063,0.5227551520511754784539479,0.5434383024128103634441936,0.5637966482266180839144308,0.5838180216287630895500389,0.6034904561585486242035732,0.6228021939105849107615396,0.6417416925623075571535249,0.6602976322726460521059468,0.6784589224477192593677557,0.6962147083695143323850866,0.7135543776835874133438599,0.7304675667419088064717369,0.7469441667970619811698824,0.7629743300440947227797691,0.7785484755064119668504941,0.7936572947621932902433329,0.8082917575079136601196422,0.8224431169556438424645942,0.8361029150609068471168753,0.8492629875779689691636001,0.8619154689395484605906323,0.8740527969580317986954180,0.8856677173453972174082924,0.8967532880491581843864474,0.9073028834017568139214859,0.9173101980809605370364836,0.9267692508789478433346245,0.9356743882779163757831268,0.9440202878302201821211114,0.9518019613412643862177963,0.9590147578536999280989185,0.9656543664319652686458290,0.9717168187471365809043384,0.9771984914639073871653744,0.9820961084357185360247656,0.9864067427245862088712355,0.9901278184917343833379303,0.9932571129002129353034372,0.9957927585349811868641612,0.9977332486255140198821574,0.9990774599773758950119878,0.9998248879471319144736081};
//static double w128[64] = {0.0244461801962625182113259,0.0244315690978500450548486,0.0244023556338495820932980,0.0243585572646906258532685,0.0243002001679718653234426,0.0242273192228152481200933,0.0241399579890192849977167,0.0240381686810240526375873,0.0239220121367034556724504,0.0237915577810034006387807,0.0236468835844476151436514,0.0234880760165359131530253,0.0233152299940627601224157,0.0231284488243870278792979,0.0229278441436868469204110,0.0227135358502364613097126,0.0224856520327449668718246,0.0222443288937997651046291,0.0219897106684604914341221,0.0217219495380520753752610,0.0214412055392084601371119,0.0211476464682213485370195,0.0208414477807511491135839,0.0205227924869600694322850,0.0201918710421300411806732,0.0198488812328308622199444,0.0194940280587066028230219,0.0191275236099509454865185,0.0187495869405447086509195,0.0183604439373313432212893,0.0179603271850086859401969,0.0175494758271177046487069,0.0171281354231113768306810,0.0166965578015892045890915,0.0162550009097851870516575,0.0158037286593993468589656,0.0153430107688651440859909,0.0148731226021473142523855,0.0143943450041668461768239,0.0139069641329519852442880,0.0134112712886163323144890,0.0129075627392673472204428,0.0123961395439509229688217,0.0118773073727402795758911,0.0113513763240804166932817,0.0108186607395030762476596,0.0102794790158321571332153,0.0097341534150068058635483,0.0091830098716608743344787,0.0086263777986167497049788,0.0080645898904860579729286,0.0074979819256347286876720,0.0069268925668988135634267,0.0063516631617071887872143,0.0057726375428656985893346,0.0051901618326763302050708,0.0046045842567029551182905,0.0040162549837386423131943,0.0034255260409102157743378,0.0028327514714579910952857,0.0022382884309626187436221,0.0016425030186690295387909,0.0010458126793403487793129,0.0004493809602920903763943};

typedef struct tagGLAW{
    int numgrid; // # of nodes
    double* u; // nodes
    double* w; // weights
} GLAW;
static GLAW glaw = {64, x64, w64};
//static GLAW glaw = {96, x96, w96};
//static GLAW glaw = {128, x128, w128};

complex<double> one(1.0, 0.0), zero(0.0, 0.0), two(2.0, 0.0), i(0.0, 1.0);
const double pi = 4.0*atan(1.0), lb = 0.0, ub = 200, Q = 0.5*(ub - lb), P = 0.5*(ub + lb);

// market parameters: you may change the number of observations by modifying the size of T and K
struct mktpara{
    double S;
    double r;
    double T[40];
    double K[40];
};

// integrands for Heston pricer:
struct tagMN{
    double M1;
    double N1;
    double M2;
    double N2;
};

// return integrands (real-valued) for Heston pricer
tagMN HesIntMN(double u, double a, double b, double c, double rho, double v0,
               double K, double T, double S, double r) {
    tagMN MNbas;

    double csqr = pow(c,2);
    double PQ_M = P+Q*u, PQ_N = P-Q*u;

    complex<double> imPQ_M = i*PQ_M;
    complex<double> imPQ_N = i*PQ_N;
    complex<double> _imPQ_M = i*(PQ_M-i);
    complex<double> _imPQ_N = i*(PQ_N-i);

    complex<double> h_M = pow(K, -imPQ_M)/imPQ_M;
    complex<double> h_N = pow(K, -imPQ_N)/imPQ_N;

    double x0 = log(S) + r*T;
    // kes = a-i*c*rho*u1;
    double tmp = c*rho;
    complex<double> kes_M1 = a - tmp*_imPQ_M;
    complex<double> kes_N1 = a - tmp*_imPQ_N;
    complex<double> kes_M2 = kes_M1 + tmp;
    complex<double> kes_N2 = kes_N1 + tmp;

    // m = i*u1 + pow(u1,2);
    complex<double> m_M1 = imPQ_M + one + pow(PQ_M-i, 2); // m_M1 = (PQ_M-i)*i + pow(PQ_M-i, 2);
    complex<double> m_N1 = imPQ_N + one + pow(PQ_N-i, 2); // m_N1 = (PQ_N-i)*i + pow(PQ_N-i, 2);
    complex<double> m_M2 = imPQ_M + pow(PQ_M-zero*i, 2);
    complex<double> m_N2 = imPQ_N + pow(PQ_N-zero*i, 2);

    // d = sqrt(pow(kes,2) + m*pow(c,2));
    complex<double> d_M1 = sqrt(pow(kes_M1,2) + m_M1*csqr);
    complex<double> d_N1 = sqrt(pow(kes_N1,2) + m_N1*csqr);
    complex<double> d_M2 = sqrt(pow(kes_M2,2) + m_M2*csqr);
    complex<double> d_N2 = sqrt(pow(kes_N2,2) + m_N2*csqr);

    // g = exp(-a*b*rho*T*u1*i/c);
    double tmp1 = -a*b*rho*T/c;
    tmp = exp(tmp1);
    complex<double> g_M2 = exp(tmp1*imPQ_M);
    complex<double> g_N2 = exp(tmp1*imPQ_N);
    complex<double> g_M1 = g_M2*tmp;
    complex<double> g_N1 = g_N2*tmp;

    // alp, calp, salp
    tmp = 0.5*T;
    complex<double> alpha = d_M1*tmp;
    complex<double> calp_M1 = cosh(alpha);
    complex<double> salp_M1 = sinh(alpha);

    alpha = d_N1*tmp;
    complex<double> calp_N1 = cosh(alpha);
    complex<double> salp_N1 = sinh(alpha);

    alpha = d_M2*tmp;
    complex<double> calp_M2 = cosh(alpha);
    complex<double> salp_M2 = sinh(alpha);

    alpha = d_N2*tmp;
    complex<double> calp_N2 = cosh(alpha);
    complex<double> salp_N2 = sinh(alpha);

    // A2 = d*calp + kes*salp;
    complex<double> A2_M1 = d_M1*calp_M1 + kes_M1*salp_M1;
    complex<double> A2_N1 = d_N1*calp_N1 + kes_N1*salp_N1;
    complex<double> A2_M2 = d_M2*calp_M2 + kes_M2*salp_M2;
    complex<double> A2_N2 = d_N2*calp_N2 + kes_N2*salp_N2;

    // A1 = m*salp;
    complex<double> A1_M1 = m_M1*salp_M1;
    complex<double> A1_N1 = m_N1*salp_N1;
    complex<double> A1_M2 = m_M2*salp_M2;
    complex<double> A1_N2 = m_N2*salp_N2;

    // A = A1/A2;
    complex<double> A_M1 = A1_M1/A2_M1;
    complex<double> A_N1 = A1_N1/A2_N1;
    complex<double> A_M2 = A1_M2/A2_M2;
    complex<double> A_N2 = A1_N2/A2_N2;

    // characteristic function: y1 = exp(i*x0*u1) * exp(-v0*A) * g * exp(2*a*b/pow(c,2)*D)
    tmp = 2*a*b/csqr;
    double halft = 0.5*T;
    complex<double> D_M1 = log(d_M1) + (a - d_M1)*halft - log((d_M1 + kes_M1)*0.5 + (d_M1 - kes_M1)*0.5*exp(-d_M1*T));
    complex<double> D_M2 = log(d_M2) + (a - d_M2)*halft - log((d_M2 + kes_M2)*0.5 + (d_M1 - kes_M2)*0.5*exp(-d_M2*T));
    complex<double> D_N1 = log(d_N1) + (a - d_N1)*halft - log((d_N1 + kes_N1)*0.5 + (d_N1 - kes_N1)*0.5*exp(-d_N1*T));
    complex<double> D_N2 = log(d_N2) + (a - d_N2)*halft - log((d_N2 + kes_N2)*0.5 + (d_N2 - kes_N2)*0.5*exp(-d_N2*T));

    MNbas.M1 = real(h_M*exp(x0*_imPQ_M - v0*A_M1 + tmp * D_M1) * g_M1);
    MNbas.N1 = real(h_N*exp(x0*_imPQ_N - v0*A_N1 + tmp * D_N1) * g_N1);
    MNbas.M2 = real(h_M*exp(x0*imPQ_M - v0*A_M2 + tmp * D_M2) * g_M2);
    MNbas.N2 = real(h_N*exp(x0*imPQ_N - v0*A_N2 + tmp * D_N2) * g_N2);

    return MNbas;
}

// Heston pricer: (parameter, observation, dim_p, dim_x, arguments)
void fHes(double *p, double *x, int m, int n, void *data)
{
    int l;

    // retrieve market parameters
    struct mktpara *dptr;
    dptr=(struct mktpara *)data;
    double S = dptr->S;
    double r = dptr->r;

    // retrieve model parameters
    double a = p[0];
    double b = p[1];
    double c = p[2];
    double rho = p[3];
    double v0 = p[4];

    // numerical integral settings
    int NumGrids = glaw.numgrid;
    NumGrids = (NumGrids+1)>>1;
    double *u = glaw.u;
    double *w = glaw.w;

    for (l=0; l<n; ++l) {
        double K = dptr->K[l];
        double T = dptr->T[l];
        double disc = exp(-r*T);
        double tmp = 0.5*(S-K*disc);
        disc = disc/pi;
        double Y1 = 0.0, Y2 = 0.0;

        for (int j=0; j< NumGrids; j++) {

            tagMN MN = HesIntMN(u[j],a, b, c, rho, v0, K, T, S, r);

            double M1 = MN.M1;
            double N1 = MN.N1;
            double M2 = MN.M2;
            double N2 = MN.N2;

            Y1 += w[j]*(M1+N1);
            Y2 += w[j]*(M2+N2);
        }

        double Qv1 = Q*Y1;
        double Qv2 = Q*Y2;
        double pv = tmp + disc*(Qv1-K*Qv2);
        x[l] = pv;
    }
}

// integrands for Jacobian


struct mktpara market;

/* set up parameters for this simulated annealing run */

/* how many points do we try before stepping */
#define N_TRIES NULL

/* how many iterations for each T? */
#define ITERS_FIXED_T 1

/* max step size in random walk */
// #define STEP_SIZE 0.005
//
// /* initial temperature */
// #define T_INITIAL 0.00000008
//
// /* damping factor for temperature */
// #define MU_T 1.000005
// #define T_MIN 0.00000006

#define STEP_SIZE 0.09
double step_cool = 1.000001;
double total_cool = step_cool;

/* initial temperature */
#define T_INITIAL 0.08

/* damping factor for temperature */
#define MU_T 1.0004
#define T_MIN 0.0004


#define BOLTZMANN_K 1
//
// double step_cool = 1.000005;
// double total_cool = step_cool;
gsl_siman_params_t params
  = {N_TRIES, ITERS_FIXED_T, STEP_SIZE,
     BOLTZMANN_K, T_INITIAL, MU_T, T_MIN};

static int m = 5;
static int n = 40; // # of observations (consistent with the struct mktpara)
// you may set up your optimal model parameters here:
// set optimal model parameters  | Corresponding model paramater |  Meaning
double a = 3.0;               // kappa                           |  mean reversion rate
double b = 0.10;              // v_infinity                      |  long term variance
double c = 0.25;              // sigma                           |  variance of volatility
double r = -0.8;            // rho                             |  correlation between spot and volatility
double v = 0.08;             // v0                              |  initial variance

double x[40];
double tempx[40];
int itercount = 0;
int fhesCount = 0;

double E1(void *xp){
  fhesCount++;
  double* p = ((double *) xp);
  double ex[n];
  // fhes computes the market observatoins with p and puts the market values into ex;
  // printf("inside E: %p\n", p);
  *(p) = 5 * pow(sin(*(p)), 2);
  *(p + 1) = pow(sin(*(p + 1)), 2);
  *(p + 2) = pow(sin(*(p + 2)), 2);
  *(p + 3) = -pow(sin(*(p + 3)), 2);
  *(p + 4) = pow(sin(*(p + 4)), 2);

  fHes(p, ex, m, n, (void *) &market);
  *(p) = asin(sqrt(*(p) * 0.2));;
  *(p + 1) = asin(sqrt(*(p + 1)));
  *(p + 2) = asin(sqrt(*(p + 2)));
  *(p + 3) = asin(sqrt(*(p + 3) * -1));
  *(p + 4) = asin(sqrt(*(p + 4)));

  double sum = 0;
  for (int i = 0; i<n; i++){
    sum += (x[i] - ex[i]) * (x[i] - ex[i]);
  }
  return sum;
}

std::ofstream iterations;
void S1(const gsl_rng * r, void *p, double step_size)
{
  itercount++;
  iterations << itercount << " ";
  double *old_p = ((double *) p);
  double new_p[m];
  *(old_p) = 5 * pow(sin(*(old_p)), 2);
  *(old_p + 1) = pow(sin(*(old_p + 1)), 2);
  *(old_p + 2) = pow(sin(*(old_p + 2)), 2);
  *(old_p + 3) = -pow(sin(*(old_p + 3)), 2);
  *(old_p + 4) = pow(sin(*(old_p + 4)), 2);
  double u = (gsl_rng_uniform(r) - 0.5);
  *(new_p) = *(old_p) + 5 * (step_size/total_cool) * u;
  u = (gsl_rng_uniform(r) - 0.5);
  *(new_p + 1) = *(old_p + 1) + (step_size/total_cool) * u;
  u = (gsl_rng_uniform(r) - 0.5);
  *(new_p + 2) = *(old_p + 2) + (step_size/total_cool) * u;
  u = (gsl_rng_uniform(r) - 0.5);
  *(new_p + 3) = *(old_p + 3) + (step_size/total_cool) * u;
  u = (gsl_rng_uniform(r) - 0.5);
  *(new_p + 4) = *(old_p + 4) + (step_size/total_cool) * u;



  if(*(new_p) > 5 || *(new_p) < 0.5){
    u = gsl_rng_uniform(r);
    *(new_p) = u * 4.5 + 0.5;
  }
  if(*(new_p + 1) > 0.95 || *(new_p + 1) < 0.05){
    u = gsl_rng_uniform(r);
    *(new_p + 1) = u * 0.9 + 0.05;
  }
  if(*(new_p + 2) > 0.95 || *(new_p + 2) < 0.05){
    u = gsl_rng_uniform(r);
    *(new_p + 2) = u * 0.9 + 0.05;
  }
  if(*(new_p + 3) > -0.1 || *(new_p + 3) < -0.9){
    u = gsl_rng_uniform(r);
    *(new_p + 3) = u* -0.8 + -0.1;
  }
  if(*(new_p + 4) > 0.95 || *(new_p + 4) < 0.05){
    u = gsl_rng_uniform(r);
    *(new_p) = u * 0.9 + 0.05;
  }


  total_cool = step_cool * total_cool;
  memcpy(old_p, &new_p, sizeof(*(new_p)) * m);
  *(old_p) = asin(sqrt(*(old_p) * 0.2));;
  *(old_p + 1) = asin(sqrt(*(old_p + 1)));
  *(old_p + 2) = asin(sqrt(*(old_p + 2)));
  *(old_p + 3) = asin(sqrt(*(old_p + 3) * -1));
  *(old_p + 4) = asin(sqrt(*(old_p + 4)));

}

std::ofstream kappa;
std::ofstream vbar;
std::ofstream sigma;
std::ofstream rho;
std::ofstream v0;
std::ofstream cost;
void P1(void *xp)
{
  double *p = ((double *) xp);

  *(p) = 5 * pow(sin(*(p)), 2);
  *(p + 1) = pow(sin(*(p + 1)), 2);
  *(p + 2) = pow(sin(*(p + 2)), 2);
  *(p + 3) = -pow(sin(*(p + 3)), 2);
  *(p + 4) = pow(sin(*(p + 4)), 2);
  kappa << sqrt((*(p) - a)*(*(p) - a))/a << " ";
  vbar << sqrt((*(p + 1) -b) * (*(p + 1) -b))/b << " ";
  sigma << sqrt((*(p + 2) -c) * (*(p + 2) -c))/c << " ";
  rho << sqrt((*(p + 3) -r)*(*(p + 3) -r))/sqrt(r*r) << " ";
  v0 << sqrt((*(p + 4) -v)*(*(p + 4) -v))/v<< " ";
  fHes(p, tempx, m, n, (void *) &market);
  printf("  %f ", *(p));
  printf("  %f ", *(p+1));
  printf("  %f ", *(p+2));
  printf("  %f ", *(p+3));
  printf("  %f ", *(p+4));
  double sum = 0;
  for (int i = 0; i<n; i++){
    sum += (x[i] - tempx[i]) * (x[i] - tempx[i]);
  }
  cost << sum << " ";
  *(p) = asin(sqrt(*(p) * 0.2));;
  *(p + 1) = asin(sqrt(*(p + 1)));
  *(p + 2) = asin(sqrt(*(p + 2)));
  *(p + 3) = asin(sqrt(*(p + 3) * -1));
  *(p + 4) = asin(sqrt(*(p + 4)));

}

int main() {
    double karr[] = {
        0.9371, 0.8603, 0.8112, 0.7760, 0.7470, 0.7216, 0.6699, 0.6137,
        0.9956, 0.9868, 0.9728, 0.9588, 0.9464, 0.9358, 0.9175, 0.9025,
        1.0427, 1.0463, 1.0499, 1.0530, 1.0562, 1.0593, 1.0663, 1.0766,
        1.2287, 1.2399, 1.2485, 1.2659, 1.2646, 1.2715, 1.2859, 1.3046,
        1.3939, 1.4102, 1.4291, 1.4456, 1.4603, 1.4736, 1.5005, 1.5328};

    // array of expiries
    // ------ where is this from too? log price domain
    double tarr[] = {0.119047619047619, 0.238095238095238,	0.357142857142857, 0.476190476190476,	0.595238095238095, 0.714285714285714, 1.07142857142857, 1.42857142857143,
        0.119047619047619	,0.238095238095238, 0.357142857142857, 0.476190476190476, 0.595238095238095, 0.714285714285714	,1.07142857142857, 1.42857142857143	,
        0.119047619047619, 	0.238095238095238,	0.357142857142857,	0.476190476190476,	0.595238095238095,	0.714285714285714,	1.07142857142857,	1.42857142857143,
        0.119047619047619,	0.238095238095238,	0.357142857142857,	0.476190476190476	,0.595238095238095,	0.714285714285714,	1.07142857142857,	1.42857142857143,
        0.119047619047619,	0.238095238095238	,0.357142857142857,	0.476190476190476,	0.595238095238095,	0.714285714285714,	1.07142857142857,	1.42857142857143};

    // strikes and expiries
    for (int j=0; j<n; ++j) {
        market.K[j] = karr[j];
        market.T[j] = tarr[j];
    }
    // spot and interest rate
    market.S = 1.0;
    market.r = 0.02;

    double pstar[5];
    pstar[0] = a; pstar[1] = b; pstar[2] = c; pstar[3] = r; pstar[4] = v;

    fHes(pstar, x, m, n, (void *) &market);

    // >>> Enter calibrating routine >>>

    // algorithm parameters
    gsl_rng_default_seed = 0;
    const gsl_rng_type * T;
    gsl_rng * r;
    T = gsl_rng_default;
    r = gsl_rng_alloc(T);

    // you may set up your initial point here:
    double p[m];
    p[0] = asin(sqrt(2 * 0.2));
    p[1] = asin(sqrt(0.2));
    p[2] = asin(sqrt(0.3));
    p[3] = asin(sqrt(-1 *-0.6));
    p[4] = asin(sqrt(0.2));
    *(p) = 5 * pow(sin(*(p)), 2);
    *(p + 1) = pow(sin(*(p + 1)), 2);
    *(p + 2) = pow(sin(*(p + 2)), 2);
    *(p + 3) = -pow(sin(*(p + 3)), 2);
    *(p + 4) = pow(sin(*(p + 4)), 2);
    double temp[40];
    fHes(p, temp, m, n, (void *) &market);
    std::ofstream costdifinit;
    costdifinit.open("costdifinit.txt");
    for (int i = 0; i <40; i++){
      printf("main: %f\n", temp[i]);
      costdifinit << sqrt(pow(x[i] - temp[i],2)) << " ";
    }
    *(p) = asin(sqrt(*(p) * 0.2));;
    *(p + 1) = asin(sqrt(*(p + 1)));
    *(p + 2) = asin(sqrt(*(p + 2)));
    *(p + 3) = asin(sqrt(*(p + 3) * -1));
    *(p + 4) = asin(sqrt(*(p + 4)));
    cout << "\r-------- -------- -------- Heston Model Calibrator -------- -------- --------"<<endl;
    cout << "Parameters:" << "\t         kappa"<<"\t     vinf"<< "\t       vov"<< "\t      rho" << "\t     v0"<<endl;
    cout << "\r Initial point:" << "\t"  << scientific << setprecision(8) << p[0]<< "\t" << p[1]<< "\t"<< p[2]<< "\t"<< p[3]<< "\t"<< p[4] << endl;
    // Calibrate using analytical gradient
    // printf("%p\n", p);
    // printf("%p\n", &p);
    kappa.open("kappa.txt");
    vbar.open("vbar.txt");
    sigma.open("sigma.txt");
    rho.open("rho.txt");
    v0.open("v0.txt");
    cost.open("cost.txt");
    iterations.open("iterations.txt");
    double start_s = clock();

    gsl_siman_solve(r, &p, E1, S1, NULL, P1,
                    NULL, NULL, NULL,
                    sizeof(double)*m, params);
    // gsl_siman_solve(r, &p, E1, S1, NULL, P1,
    //                 NULL, NULL, NULL,
    //                 sizeof(double)*m, params);
    double stop_s = clock();
    *(p) = 5 * pow(sin(*(p)), 2);
    *(p + 1) = pow(sin(*(p + 1)), 2);
    *(p + 2) = pow(sin(*(p + 2)), 2);
    *(p + 3) = -pow(sin(*(p + 3)), 2);
    *(p + 4) = pow(sin(*(p + 4)), 2);
    fHes(p, temp, m, n, (void *) &market);
    std::ofstream costdif;
    costdif.open("costdif.txt");
    for (int i = 0; i <40; i++){
      costdif << sqrt(pow(x[i] - temp[i],2))  << " ";
    }
    *(p) = asin(sqrt(*(p) * 0.2));;
    *(p + 1) = asin(sqrt(*(p + 1)));
    *(p + 2) = asin(sqrt(*(p + 2)));
    *(p + 3) = asin(sqrt(*(p + 3) * -1));
    *(p + 4) = asin(sqrt(*(p + 4)));
    printf("fhesCount %d\n", fhesCount);
    printf("time: %f\n", (stop_s - start_s)/CLOCKS_PER_SEC);


    return 0;
} // The End
